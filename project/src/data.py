"""Loading the Instacart tables and turning orders into compact baskets."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("INSTACART_DATA", Path(__file__).resolve().parent.parent / "data"))

REQUIRED_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]

ORDER_DTYPES = {
    "order_id": "int32",
    "user_id": "int32",
    "eval_set": "category",
    "order_number": "int16",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
}
ORDER_PRODUCT_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}


def check_data(data_dir=DATA_DIR):
    missing = [f for f in REQUIRED_FILES if not (Path(data_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {', '.join(missing)}. "
            "Download the Instacart data (see README) and unzip the CSVs into data/."
        )


def load_tables(data_dir=DATA_DIR):
    """Read all six CSVs. Returns a dict of DataFrames; products include aisle/department names."""
    data_dir = Path(data_dir)
    check_data(data_dir)
    orders = pd.read_csv(data_dir / "orders.csv", dtype=ORDER_DTYPES)
    prior = pd.read_csv(data_dir / "order_products__prior.csv", dtype=ORDER_PRODUCT_DTYPES)
    train = pd.read_csv(data_dir / "order_products__train.csv", dtype=ORDER_PRODUCT_DTYPES)
    products = (
        pd.read_csv(data_dir / "products.csv")
        .merge(pd.read_csv(data_dir / "aisles.csv"), on="aisle_id")
        .merge(pd.read_csv(data_dir / "departments.csv"), on="department_id")
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    return {"orders": orders, "prior": prior, "train": train, "products": products}


@dataclass
class Baskets:
    """All baskets in CSR form: the items of basket b are items[indptr[b]:indptr[b + 1]].

    Two flat int arrays instead of millions of Python lists keeps 32M rows at
    ~150 MB and lets numba loop over the baskets directly.
    """

    indptr: np.ndarray
    items: np.ndarray
    order_ids: np.ndarray

    def __len__(self):
        return len(self.order_ids)

    def __getitem__(self, b):
        return self.items[self.indptr[b]:self.indptr[b + 1]]

    def sizes(self):
        return np.diff(self.indptr)

    def head(self, n):
        """The first n baskets, e.g. for testing against a brute-force count."""
        return Baskets(self.indptr[: n + 1], self.items[: self.indptr[n]], self.order_ids[:n])


def to_baskets(order_products):
    df = order_products.sort_values(["order_id", "add_to_cart_order"])
    order_ids, starts = np.unique(df["order_id"].to_numpy(), return_index=True)
    indptr = np.append(starts, len(df)).astype(np.int64)
    return Baskets(indptr, df["product_id"].to_numpy(dtype=np.int32), order_ids.astype(np.int32))


def next_basket_truth(orders, order_products_train):
    """{user_id: set of products in their held-out next order} for the ~131k users with a 'train' order."""
    train_orders = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]
    df = order_products_train.merge(train_orders, on="order_id")
    return df.groupby("user_id")["product_id"].apply(set).to_dict()


def load_baskets(data_dir=DATA_DIR, file="order_products__prior.csv", chunksize=2_000_000, cache=True):
    """Baskets read from the CSV in chunks, without loading the whole table into pandas.

    Only order_id and product_id are kept (8 bytes per row instead of ~70 for a full DataFrame).
    Each basket is the set of unique product ids of one order, sorted by product id.
    The arrays are cached in data/derived/, so later calls take under a second.
    """
    data_dir = Path(data_dir)
    cache_file = data_dir / "derived" / (Path(file).stem + "_baskets.npz")
    if cache and cache_file.exists():
        z = np.load(cache_file)
        return Baskets(z["indptr"], z["items"], z["order_ids"])
    check_data(data_dir)
    orders, products = [], []
    for chunk in pd.read_csv(data_dir / file, usecols=["order_id", "product_id"],
                             dtype={"order_id": "int32", "product_id": "int32"}, chunksize=chunksize):
        orders.append(chunk["order_id"].to_numpy())
        products.append(chunk["product_id"].to_numpy())
    order_col, product_col = np.concatenate(orders), np.concatenate(products)
    del orders, products
    by_order = np.lexsort((product_col, order_col))
    order_col, product_col = order_col[by_order], product_col[by_order]
    # a product listed twice in one order counts once
    keep = np.ones(len(order_col), bool)
    keep[1:] = (order_col[1:] != order_col[:-1]) | (product_col[1:] != product_col[:-1])
    order_col, product_col = order_col[keep], product_col[keep]
    order_ids, starts = np.unique(order_col, return_index=True)
    baskets = Baskets(np.append(starts, len(order_col)).astype(np.int64), product_col, order_ids.astype(np.int32))
    if cache:
        cache_file.parent.mkdir(exist_ok=True)
        np.savez(cache_file, indptr=baskets.indptr, items=baskets.items, order_ids=baskets.order_ids)
    return baskets


def user_split(users, seed=0):
    """Split the users with a held-out order: {"train": 40%, "validation": 10%, "test": 50%} of user ids.

    Models are fitted on train (and validation once tuned); choices such as rule settings are
    tuned on validation; test is used once, for the final numbers.
    """
    users = np.array(sorted(users))
    r = np.random.default_rng(seed).random(len(users))
    return {"train": users[r < 0.4], "validation": users[(r >= 0.4) & (r < 0.5)], "test": users[r >= 0.5]}
