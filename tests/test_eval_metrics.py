import unittest

import math

from reviewproof.eval.metrics import (cohen_kappa, hit_at_k, mrr_at_k, ndcg_at_k, ndcg_graded_at_k,
                                     recall_at_k, three_state)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "x"}


class RetrievalMetricTests(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(RANKED, RELEVANT, 5), 2 / 3)
        self.assertAlmostEqual(recall_at_k(RANKED, RELEVANT, 2), 1 / 3)

    def test_hit(self):
        self.assertEqual((hit_at_k(RANKED, RELEVANT, 1), hit_at_k(RANKED, RELEVANT, 2)), (0.0, 1.0))

    def test_mrr(self):
        self.assertEqual(mrr_at_k(RANKED, RELEVANT, 10), 0.5)
        self.assertEqual(mrr_at_k(RANKED, {"z"}, 10), 0.0)
        self.assertEqual(mrr_at_k(RANKED, {"e"}, 3), 0.0)

    def test_ndcg(self):
        self.assertAlmostEqual(ndcg_at_k(RANKED, RELEVANT, 5), 0.477623, places=5)
        self.assertEqual(ndcg_at_k(["b"], {"b"}, 5), 1.0)

    def test_ndcg_graded(self):
        self.assertAlmostEqual(ndcg_graded_at_k(["y", "x"], {"x": 2, "y": 1}, 5),
                               (1 / 1 + 2 / math.log2(3)) / (2 / 1 + 1 / math.log2(3)))
        self.assertEqual(ndcg_graded_at_k(["x", "y"], {"x": 2, "y": 1, "z": 0}, 5), 1.0)

    def test_empty_relevant_raises(self):
        for metric in (recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k):
            with self.assertRaises(ValueError):
                metric(RANKED, set(), 5)


if __name__ == "__main__":
    unittest.main()


class AgreementMetricTests(unittest.TestCase):
    def test_cohen_kappa_known_values(self):
        self.assertEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)
        self.assertIsNone(cohen_kappa(["a", "a"], ["a", "a"]))
        self.assertIsNone(cohen_kappa([], []))

    def test_three_state_confusion_excludes_service_failures(self):
        rows = [{"status": "answered", "service_status": "ok", "answerability_gold": "answered"},
                {"status": "partial", "service_status": "ok", "answerability_gold": "answered"},
                {"status": "unanswerable", "service_status": "ok", "answerability_gold": "unanswerable"},
                {"status": None, "service_status": "failed", "answerability_gold": "partial"}]
        m = three_state(rows)
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["confusion"]["answered"], {"answered": 1, "partial": 1, "unanswerable": 0})
        self.assertEqual(m["confusion"]["partial"], {"answered": 0, "partial": 0, "unanswerable": 0})
        self.assertAlmostEqual(m["agreement"]["mean"], 2 / 3)
