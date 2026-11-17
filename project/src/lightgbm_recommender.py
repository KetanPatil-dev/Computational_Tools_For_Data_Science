"""LightGBM for next-basket recommendation, using only prior-order features."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


FEATURES = [
    "user_orders", "user_mean_basket_size", "user_unique_products",
    "purchase_count", "purchase_rate", "orders_since_last", "order_span",
    "in_last_order", "product_orders", "product_users", "product_order_share",
    "aisle_id", "department_id",
]
CATEGORICAL = ["aisle_id", "department_id"]


def product_features(orders, prior, users):
    """Product statistics fitted on these users' prior orders only.

    Fit once on the training users and reuse for validation and test. The row
    order also defines the popularity fallback (ties go to smaller product IDs).
    """
    o = orders.loc[(orders["eval_set"] == "prior") & orders["user_id"].isin(users),
                   ["order_id", "user_id"]]
    history = prior[["order_id", "product_id"]].merge(o, on="order_id")
    stats = history.groupby("product_id").agg(
        product_orders=("order_id", "size"), product_users=("user_id", "nunique"),
    ).reset_index()
    if stats.empty:
        raise ValueError("No prior purchases for the training users")
    stats["product_order_share"] = stats["product_orders"] / len(o)
    return stats.sort_values(["product_orders", "product_id"], ascending=[False, True],
                             ignore_index=True)


def make_features(orders, prior, products, users, stats, n_popular=20):
    """One row per candidate (user, product), without looking at the next order.

    Candidates are all products the user has bought plus n_popular products
    from the training-history popularity list. This allows some new products
    without constructing a user x catalogue cross join. The model ranks these
    products; the popularity fallback can fill short lists at recommendation time.

    Missing purchase counts are zero. Recency/span are NaN for unseen products,
    which LightGBM handles directly. User and product IDs are keys, not features.
    """
    if n_popular < 0:
        raise ValueError("n_popular must be non-negative")
    users = np.unique(np.asarray(users, dtype=np.int32))
    o = orders.loc[(orders["eval_set"] == "prior") & orders["user_id"].isin(users),
                   ["order_id", "user_id", "order_number"]]
    history = prior[["order_id", "product_id"]].merge(o, on="order_id")
    own = history.groupby(["user_id", "product_id"]).agg(
        purchase_count=("order_id", "size"), first_order=("order_number", "min"),
        last_order=("order_number", "max"),
    ).reset_index()
    user_stats = o.groupby("user_id").agg(
        user_orders=("order_id", "size"), last_prior_order=("order_number", "max"),
    )
    user_stats["user_mean_basket_size"] = history.groupby("user_id").size() / user_stats["user_orders"]
    user_stats["user_unique_products"] = own.groupby("user_id").size()
    popular = stats.sort_values(["product_orders", "product_id"], ascending=[False, True])
    popular = popular["product_id"].head(n_popular).to_numpy()
    extra = pd.DataFrame({"user_id": np.repeat(users, len(popular)),
                          "product_id": np.tile(popular, len(users))})
    candidates = pd.concat([own[["user_id", "product_id"]], extra], ignore_index=True)
    candidates = candidates.drop_duplicates(ignore_index=True)
    features = (candidates.merge(own, on=["user_id", "product_id"], how="left")
                .merge(user_stats, on="user_id", how="left")
                .merge(stats, on="product_id", how="left", validate="many_to_one")
                .merge(products[["product_id", "aisle_id", "department_id"]],
                       on="product_id", how="left", validate="many_to_one"))
    zero_fill = ["purchase_count", "user_orders", "user_mean_basket_size",
                 "user_unique_products", "product_orders", "product_users", "product_order_share"]
    features[zero_fill] = features[zero_fill].fillna(0)
    features["purchase_rate"] = features["purchase_count"] / features["user_orders"].replace(0, np.nan)
    features["purchase_rate"] = features["purchase_rate"].fillna(0)
    features["orders_since_last"] = features["last_prior_order"] - features["last_order"]
    features["order_span"] = features["last_order"] - features["first_order"]
    features["in_last_order"] = features["orders_since_last"].eq(0).astype(np.float32)
    features[CATEGORICAL] = features[CATEGORICAL].fillna(-1).astype(np.int32)
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    features[numeric] = features[numeric].astype(np.float32)
    features[["user_id", "product_id"]] = features[["user_id", "product_id"]].astype(np.int32)
    return features[["user_id", "product_id", *FEATURES]].sort_values(
        ["user_id", "product_id"], ignore_index=True)


def candidate_labels(features, truth):
    """Binary next-order labels, kept separate from feature construction.

    A missing user's target is an error, not an all-negative training example.
    Targets outside the candidate set remain in truth for the final metrics.
    """
    return np.fromiter((p in truth[u] for u, p in
                        features[["user_id", "product_id"]].itertuples(index=False, name=None)),
                       dtype=np.int8, count=len(features))


def fit_lightgbm(train_features, train_truth, validation_features, validation_truth,
                 seed=0, stopping_rounds=30, **params):
    """Fit binary purchase prediction; early stopping uses validation log loss.

    Compare model settings using evaluate_at_k on validation recommendations.
    Test features and labels are never passed here.
    """
    if train_features.empty or validation_features.empty:
        raise ValueError("Training and validation candidates must both be non-empty")
    if train_features["user_id"].isin(validation_features["user_id"]).any():
        raise ValueError("Training and validation users must be disjoint")
    y_train = candidate_labels(train_features, train_truth)
    y_validation = candidate_labels(validation_features, validation_truth)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training candidates must include both positive and negative labels")
    settings = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                    min_child_samples=50, colsample_bytree=0.9, reg_lambda=1.0,
                    n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    settings.update(params)
    model = LGBMClassifier(objective="binary", random_state=seed, **settings)
    model.fit(train_features[FEATURES], y_train,
              eval_set=[(validation_features[FEATURES], y_validation)],
              eval_metric="binary_logloss", categorical_feature=CATEGORICAL,
              callbacks=[early_stopping(stopping_rounds, verbose=False), log_evaluation(0)])
    return model


def recommend_lightgbm(model, features, users, k, fallback, chunk=100_000):
    """Rank purchase probabilities, breaking ties by product ID; fill from fallback.

    Prediction is chunked to avoid a second full feature matrix. Recommendations
    have the same {user_id: [product_id, ...]} format as recommend_rules.
    """
    if k <= 0 or chunk <= 0:
        raise ValueError("k and chunk must be positive")
    users = np.unique(np.asarray(users, dtype=np.int32))
    ranked = features.loc[features["user_id"].isin(users), ["user_id", "product_id"]].copy()
    scores = np.empty(len(ranked), dtype=np.float64)
    for start in range(0, len(ranked), chunk):
        batch = features.loc[ranked.index[start:start + chunk], FEATURES]
        scores[start:start + chunk] = model.predict_proba(batch)[:, 1]
    ranked["score"] = scores
    ranked = ranked.sort_values(["user_id", "score", "product_id"], ascending=[True, False, True])
    ranked = ranked.drop_duplicates(["user_id", "product_id"])
    recs = ranked.groupby("user_id").head(k).groupby("user_id")["product_id"].apply(list).to_dict()
    fallback = list(dict.fromkeys(fallback))
    for u in users:
        own = recs.get(u, [])
        for p in fallback:
            if len(own) >= k:
                break
            if p not in own:
                own.append(p)
        recs[u] = own
    return recs
