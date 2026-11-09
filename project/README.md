# A-Priori and PCY on Instacart baskets

Frequent-itemset mining on 3.2M real grocery baskets from the Instacart Online Grocery Shopping Dataset 2017, and a next-basket recommender built from the mined association rules.
A-Priori and PCY are our own implementations, compiled with numba.

This folder covers two parts of a larger project:

- **Data preparation and A-Priori** (`notebooks/01_data_apriori.ipynb`)
- **PCY, association rules and the rule-based recommender** (`notebooks/02_pcy_rules_recommender.ipynb`)

## Results

**Data**
- 3,214,874 prior baskets from 206,209 users and 49,688 products; 10.1 items per basket on average.
- Integrity checks are asserted in the notebook: each product appears at most once per order, every user's last order is their train or test order, and all IDs match.

**A-Priori**
- Matches a toy example worked out by hand and a brute-force count on 20,000 real baskets.
- At support 1e-4 it finds 57,035 frequent pairs in about 2 s. The triangular matrix needs 239 MB.

**PCY**
- Returns exactly A-Priori's pairs and counts.
- Support 2e-5 (64 baskets), 428,017 frequent pairs. Memory of the last pass:

  | Buckets | PCY | Multistage (extension) |
  |---|---|---|
  | 2^18 | 417 MB | 417 MB |
  | 2^22 | 96 MB | 18 MB |
  | 2^24 | 22 MB | 10 MB |

  With 2^18 buckets every bucket is frequent, so PCY ends up counting all 34.7M pairs.

**Rule-based recommender**
- Tuned on the validation users, then evaluated once on the 65,444 test users.
- Hit-rate@10 is 50.6% and precision@10 is 0.085.

**Same basket ≠ next basket**
- Take a rule with confidence above 0.3 whose consequent is new to the user. That product appears in the user's next order only 4% of the time.
- If the user has bought the product before, the rate is 41%.
- So a rule's confidence measures the same basket, not the next one.

## What is included

**Planned work**
- Loading and checking the six files.
- Baskets: one per order, as a set of unique product IDs, read in chunks and cached.
- The train / validation / test split.
- Descriptive statistics.
- A-Priori with correctness checks.
- PCY with a bucket-size experiment and a check that it matches A-Priori.
- Association rules with support, confidence and lift.
- The rule-based recommender.
- The same-basket vs next-basket analysis.

**Extensions beyond the plan**
- Multistage and Multihash, the two refinements of PCY (`src/pcy.py`).
- Frequent triples, from a third A-Priori pass checked against brute force.

## Data

The raw CSVs are not in this repository: two of them are larger than GitHub's 100 MB file limit. To get them:

1. Download the data from https://www.kaggle.com/competitions/instacart-market-basket-analysis/data. You need to sign in and accept the competition rules.
2. Unzip these six files into `data/`:

```
data/aisles.csv  data/departments.csv  data/products.csv
data/orders.csv  data/order_products__prior.csv  data/order_products__train.csv
```

To keep the CSVs somewhere else, set `INSTACART_DATA=/path/to/csvs`.

`data/derived/` contains these outputs, which the rest of the project can use directly:

| File | Contents |
|---|---|
| `split.csv` | train / validation / test assignment of every evaluated user |
| `sample_baskets.npz` | 100k baskets for quick development; load with `Baskets(**np.load(...))` |
| `rules_support_1e-4.csv` | association rules with support, confidence and lift |
| `rule_recommendations_test.csv` | the rule recommender's top-10 for every test user |

## Running

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Run `01` first. It builds and caches the baskets (about 160 MB in `data/derived/`, not committed) and saves the user split. The notebooks are saved with their outputs, so the results can be read without running anything.

| Notebook | Contents | Time |
|---|---|---|
| `01_data_apriori` | the six files and integrity checks, descriptive statistics, chunked basket loading, train/validation/test split, A-Priori (toy example, brute-force check, full data), frequent triples | ~1 min |
| `02_pcy_rules_recommender` | PCY step by step on the toy example, PCY = A-Priori, bucket-size experiment, association rules, rule-based recommender (tuned on validation, evaluated once on test), same-basket vs next-basket analysis | ~6 min |

## Code

| Path | Contents |
|---|---|
| `src/data.py` | loading, chunked basket loader with cache (CSR arrays), held-out next baskets, user split |
| `src/apriori.py` | A-Priori (triangular matrix), frequent triples, association rules |
| `src/pcy.py` | PCY, Multistage, Multihash |
| `src/hashcount.py` | open-addressing hash table for sparse pair and triple counts |
| `src/recommend.py` | rule-based recommender, precision / recall / hit-rate@k |

## Evaluation protocol

- **Ground truth:** for each of the 131,209 users with an order in Instacart's `train` set, that order. Only the users' `prior` orders are used as input.
- **Split:** users are split at random with a fixed seed into train (40%), validation (10%, for tuning) and test (50%, used once).
- **Primary metric:** hit-rate@10, the share of users with at least one recommended product in their next order. Precision@10 and recall@10 are also reported.
