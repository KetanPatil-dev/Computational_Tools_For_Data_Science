# A-Priori, PCY and LightGBM on Instacart baskets

Frequent-itemset mining on 3.2M real grocery baskets from the Instacart Online Grocery Shopping Dataset 2017, and a next-basket recommender built from the mined association rules.
A-Priori and PCY are our own implementations, compiled with numba.
LightGBM adds a supervised next-basket recommender using the LightGBM library.

This folder covers three parts of a larger project:

- **Data preparation and A-Priori** (`notebooks/01_data_apriori.ipynb`)
- **PCY, association rules and the rule-based recommender** (`notebooks/02_pcy_rules_recommender.ipynb`)
- **LightGBM next-basket recommender** (`notebooks/03_lightgbm_recommender.ipynb`)

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

**LightGBM recommender**
- Prior-order purchase frequency, recency, user history, product popularity and product categories predict whether a candidate appears in the next order.
- Uses the saved user split and the same top-10 evaluation metrics as the rule recommender. Candidates are previously purchased products plus the 20 most popular training-history products.
- Full-data results are pending: the raw CSVs are required to run notebook 03. The feature counts, label isolation, training and ranking are checked on synthetic data.

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
- A supervised LightGBM comparison, with validation tuning, feature importance and candidate coverage.

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

Notebook 03 additionally writes `lightgbm_recommendations_test.csv` in the same
`user_id, rank, product_id` format, `lightgbm_test_metrics.csv`,
`lightgbm_validation.csv` and `lightgbm_model.txt`. These generated files are ignored by Git.
The sample basket cache alone cannot be used to train LightGBM: it lacks user histories and next-order labels.

## Running

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Run `01` first. It builds and caches the baskets (about 160 MB in `data/derived/`, not committed) and saves the user split. Notebooks 01–02 have saved outputs; notebook 03 must be run with the raw data to produce full-data results.

On macOS, LightGBM also needs OpenMP (`brew install libomp`); see the
[installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).

| Notebook | Contents | Time |
|---|---|---|
| `01_data_apriori` | the six files and integrity checks, descriptive statistics, chunked basket loading, train/validation/test split, A-Priori (toy example, brute-force check, full data), frequent triples | ~1 min |
| `02_pcy_rules_recommender` | PCY step by step on the toy example, PCY = A-Priori, bucket-size experiment, association rules, rule-based recommender (tuned on validation, evaluated once on test), same-basket vs next-basket analysis | ~6 min |
| `03_lightgbm_recommender` | feature checks, saved user split, candidate coverage, LightGBM validation tuning, feature importance, test comparison and prediction export | depends on candidate count; not yet measured on full data |

Notebook 03 needs 01's split; running 02 is optional and adds the saved rule predictions
to its final comparison. Training and validation feature frames are held in memory;
allow several GB of RAM for the full dataset. Test feature construction uses batches of 5,000 users.

Run the LightGBM checks from this folder after installing the requirements:

```bash
python -m unittest discover -s tests -v
```

## Code

| Path | Contents |
|---|---|
| `src/data.py` | loading, chunked basket loader with cache (CSR arrays), held-out next baskets, user split |
| `src/apriori.py` | A-Priori (triangular matrix), frequent triples, association rules |
| `src/pcy.py` | PCY, Multistage, Multihash |
| `src/hashcount.py` | open-addressing hash table for sparse pair and triple counts |
| `src/recommend.py` | rule-based recommender, precision / recall / hit-rate@k |
| `src/lightgbm_recommender.py` | prior-order features and candidates, binary labels, LightGBM training, top-k ranking |
| `tests/test_lightgbm_recommender.py` | feature counts, leakage checks, ranking edge cases and synthetic training |

## Evaluation protocol

- **Ground truth:** for each of the 131,209 users with an order in Instacart's `train` set, that order. Only the users' `prior` orders are used as input.
- **Split:** users are split at random with a fixed seed into train (40%), validation (10%, for tuning) and test (50%, used once).
- **Primary metric:** hit-rate@10, the share of users with at least one recommended product in their next order. Precision@10 and recall@10 are also reported.
- **LightGBM:** product statistics are fitted on training users' prior orders and frozen for validation and test. Each user's own prior history supplies their personal features. Early stopping uses validation log loss; tree size is selected by validation hit-rate@10. The chosen model stays fitted on training users, and all test users are evaluated against their full target baskets, including products outside the candidate set.
- **History pools:** notebook 02 mines rules from all users' prior baskets; LightGBM's product statistics and its popularity baseline use training users' prior baskets. The comparison shares users and targets but has this difference in global history inputs.
