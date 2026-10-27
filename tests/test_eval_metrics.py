import unittest

from reviewproof.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k

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

    def test_empty_relevant_raises(self):
        for metric in (recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k):
            with self.assertRaises(ValueError):
                metric(RANKED, set(), 5)


if __name__ == "__main__":
    unittest.main()
