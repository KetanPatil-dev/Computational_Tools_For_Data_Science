"""Rule-based next-basket recommendation and its evaluation."""
import numpy as np


def global_top(prior, k):
    """The k products bought most often overall."""
    return prior["product_id"].value_counts().index[:k].tolist()


def evaluate_at_k(recs, truth, k=10):
    """Mean precision@k, recall@k and hit-rate@k over all users in `truth`."""
    precision, recall, hit = [], [], []
    for u, actual in truth.items():
        n_hit = sum(1 for p in recs.get(u, [])[:k] if p in actual)
        precision.append(n_hit / k)
        recall.append(n_hit / len(actual))
        hit.append(n_hit > 0)
    return {
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        f"hit_rate@{k}": float(np.mean(hit)),
        "users": len(truth),
    }


def recent_products(orders, prior, users, n_last=1):
    """(user_id, product_id) of the distinct products in each user's n_last most recent prior orders."""
    o = orders.loc[(orders["eval_set"] == "prior") & orders["user_id"].isin(set(users)), ["order_id", "user_id", "order_number"]]
    last = o.groupby("user_id")["order_number"].transform("max")
    o = o[o["order_number"] > last - n_last]
    df = prior[["order_id", "product_id"]].merge(o[["order_id", "user_id"]], on="order_id")
    return df[["user_id", "product_id"]].drop_duplicates(ignore_index=True)


def recommend_rules(history, rules, users, k, fallback, score="confidence", max_rules=20, chunk=10_000):
    """Rule-based next-basket recommendation.

    Every rule a -> b whose antecedent a is in the user's history votes for b with its score
    (confidence or lift); the k products with the highest summed votes are recommended.
    Products the user bought before may be recommended. Empty positions are filled from `fallback`.
    Only each antecedent's max_rules most confident rules are used, and users are processed
    in chunks, to keep the join small.
    """
    r = (rules.sort_values("confidence", ascending=False).groupby("antecedent").head(max_rules)
         [["antecedent", "consequent", score]])
    users = np.asarray(users)
    recs = {}
    for start in range(0, len(users), chunk):
        h = history[history["user_id"].isin(set(users[start:start + chunk]))]
        m = h.merge(r, left_on="product_id", right_on="antecedent")
        votes = m.groupby(["user_id", "consequent"])[score].sum().rename("votes").reset_index()
        votes = votes.sort_values(["user_id", "votes", "consequent"], ascending=[True, False, True])
        recs.update(votes.groupby("user_id").head(k).groupby("user_id")["consequent"].apply(list).to_dict())
    for u in users:
        own = recs.get(u, [])
        recs[u] = own + [p for p in fallback if p not in own][: k - len(own)]
    return recs
