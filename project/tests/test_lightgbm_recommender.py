"""Run with: python -m unittest discover -s tests -v (from project/)."""
import unittest

import numpy as np
import pandas as pd

from src.lightgbm_recommender import (
    FEATURES, candidate_labels, fit_lightgbm, make_features,
    product_features, recommend_lightgbm,
)
from src.recommend import evaluate_at_k


def toy_tables():
    orders = pd.DataFrame([
        (11, 1, "prior", 1), (12, 1, "prior", 2), (13, 1, "train", 3),
        (21, 2, "prior", 1), (22, 2, "prior", 2), (23, 2, "train", 3),
    ], columns=["order_id", "user_id", "eval_set", "order_number"])
    prior = pd.DataFrame([(11, 1), (11, 2), (12, 1), (12, 3),
                          (21, 2), (21, 4), (22, 4)], columns=["order_id", "product_id"])
    products = pd.DataFrame({"product_id": [1, 2, 3, 4, 5],
                             "aisle_id": [1, 1, 2, 2, 3], "department_id": [1, 1, 1, 1, 2]})
    return orders, prior, products


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.orders, self.prior, self.products = toy_tables()
        self.stats = product_features(self.orders, self.prior, [1])

    def features(self, users, **kwargs):
        return make_features(self.orders, self.prior, self.products, users, self.stats, **kwargs)

    def test_features_match_hand_counts(self):
        f = self.features([1], n_popular=1).set_index("product_id")
        self.assertEqual(set(f.index), {1, 2, 3})
        self.assertEqual(f.loc[1, "purchase_count"], 2)
        self.assertEqual(f.loc[1, "purchase_rate"], 1)
        self.assertEqual(f.loc[1, "user_orders"], 2)
        self.assertEqual(f.loc[1, "user_mean_basket_size"], 2)
        self.assertEqual(f.loc[1, "user_unique_products"], 3)
        self.assertEqual(f.loc[1, "order_span"], 1)
        self.assertEqual(f.loc[2, "orders_since_last"], 1)
        self.assertEqual(f.loc[2, "in_last_order"], 0)
        self.assertEqual(f.loc[3, "in_last_order"], 1)
        self.assertEqual(f.loc[1, "product_order_share"], 1)
        self.assertNotIn("user_id", FEATURES)
        self.assertNotIn("product_id", FEATURES)

    def test_unseen_products_and_users(self):
        f = self.features([2, 3], n_popular=1).set_index(["user_id", "product_id"])
        self.assertEqual(set(f.loc[2].index), {1, 2, 4})
        self.assertEqual(f.loc[(2, 1), "purchase_count"], 0)
        self.assertTrue(np.isnan(f.loc[(2, 1), "orders_since_last"]))
        self.assertEqual(f.loc[(2, 4), "product_orders"], 0)
        self.assertEqual(f.loc[(3, 1), "user_orders"], 0)
        self.assertEqual(f.loc[(3, 1), "purchase_rate"], 0)

    def test_future_rows_do_not_change_features_or_statistics(self):
        before = self.features([1, 2])
        self.prior = pd.concat([self.prior, pd.DataFrame({"order_id": [13, 23], "product_id": [5, 5]})])
        self.orders.loc[self.orders["eval_set"] == "train", "order_number"] = 999
        pd.testing.assert_frame_equal(before, self.features([1, 2]))
        pd.testing.assert_frame_equal(self.stats, product_features(self.orders, self.prior, [1]))
        # Validation histories also cannot change training product statistics.
        self.prior = self.prior[self.prior["order_id"] != 21]
        pd.testing.assert_frame_equal(self.stats, product_features(self.orders, self.prior, [1]))

    def test_candidates_do_not_depend_on_targets(self):
        f = self.features([1], n_popular=0)
        labels = candidate_labels(f, {1: {1, 5}})
        self.assertEqual(labels.tolist(), [1, 0, 0])
        self.assertNotIn(5, f["product_id"].tolist())
        self.assertEqual(evaluate_at_k({1: [1]}, {1: {1, 5}}, 1)["recall@1"], 0.5)
        with self.assertRaises(KeyError):
            candidate_labels(f, {})

    def test_empty_candidates_and_unknown_categories(self):
        f = self.features([], n_popular=0)
        self.assertEqual(f.columns.tolist(), ["user_id", "product_id", *FEATURES])
        self.assertTrue(f.empty)
        self.products = self.products[self.products["product_id"] != 4]
        f = self.features([2], n_popular=0).set_index("product_id")
        self.assertEqual(f.loc[4, "aisle_id"], -1)

    def test_validation_guards(self):
        with self.assertRaises(ValueError):
            self.features([1], n_popular=-1)
        with self.assertRaises(ValueError):
            product_features(self.orders, self.prior, [])
        f = self.features([1])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_lightgbm(f, {1: {1}}, f, {1: {1}})
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            fit_lightgbm(f, {1: {5}}, self.features([2]), {2: {4}})


class RecommendationTests(unittest.TestCase):
    def test_ties_chunking_deduplication_and_fallback(self):
        class ConstantModel:
            def predict_proba(self, batch):
                return np.full((len(batch), 2), 0.5)

        orders, prior, products = toy_tables()
        stats = product_features(orders, prior, [1])
        f = make_features(orders, prior, products, [1, 2], stats, n_popular=1)
        # Retain non-contiguous indices, as a filtered frame would.
        f = f[f["product_id"] != 2]
        model = ConstantModel()
        recs = recommend_lightgbm(model, f, [1, 2, 3], 4, [1, 1, 2, 3, 4, 5], chunk=1)
        self.assertEqual(recs, {1: [1, 3, 2, 4], 2: [1, 4, 2, 3], 3: [1, 2, 3, 4]})
        self.assertEqual(recs, recommend_lightgbm(model, f, [1, 2, 3], 4, [1, 2, 3, 4, 5]))
        self.assertEqual(recommend_lightgbm(model, f.iloc[:0], [3], 2, [4, 5]), {3: [4, 5]})
        with self.assertRaises(ValueError):
            recommend_lightgbm(model, f, [1], 0, [1])

    def test_real_lightgbm_training_and_prediction(self):
        orders, purchases, truth = [], [], {}
        for u in range(1, 41):
            for n in range(1, 4):
                order = u * 10 + n
                orders.append((order, u, "prior", n))
                purchases.extend([(order, 1), (order, 2 + u % 3)])
            orders.append((u * 10 + 4, u, "train", 4))
            truth[u] = {1}
        orders = pd.DataFrame(orders, columns=["order_id", "user_id", "eval_set", "order_number"])
        prior = pd.DataFrame(purchases, columns=["order_id", "product_id"])
        products = toy_tables()[2]
        stats = product_features(orders, prior, range(1, 25))
        train = make_features(orders, prior, products, range(1, 25), stats)
        val = make_features(orders, prior, products, range(25, 33), stats)
        test = make_features(orders, prior, products, range(33, 41), stats)
        params = dict(n_estimators=30, min_child_samples=2, num_leaves=7, n_jobs=1)
        model = fit_lightgbm(train, truth, val, truth, stopping_rounds=5, **params)
        recs = recommend_lightgbm(model, test, range(33, 41), 2, stats["product_id"])
        self.assertTrue(all(ps[0] == 1 for ps in recs.values()))
        self.assertEqual(evaluate_at_k(recs, {u: truth[u] for u in recs}, 2)["hit_rate@2"], 1)
        self.assertEqual(model.feature_name_, FEATURES)
        self.assertGreater(model.best_iteration_, 0)
        other = fit_lightgbm(train, truth, val, truth, stopping_rounds=5, **params)
        np.testing.assert_allclose(model.predict_proba(test[FEATURES]), other.predict_proba(test[FEATURES]))


if __name__ == "__main__":
    unittest.main()
