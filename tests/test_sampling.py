import tempfile
import unittest
from collections import Counter
from pathlib import Path

from reviewproof.ingest.sampling import (
    ProductStats,
    assign_splits,
    band_for,
    scan_product_stats,
    select_sample,
)
from tests.support import write_jsonl

PER_BAND = {"large": 10, "medium": 10, "small": 5}
BAND_REVIEWS = {"large": 300, "medium": 100, "small": 10}


def make_stats(category="Cat", large=30, medium=30, small=30, questions=6):
    stats = []
    for band, n in (("large", large), ("medium", medium), ("small", small)):
        for i in range(n):
            stats.append(ProductStats(f"{category}-{band}-{i:02d}", category, BAND_REVIEWS[band], questions))
    return stats


class SamplingTests(unittest.TestCase):
    def test_band_boundaries(self):
        self.assertEqual([band_for(n) for n in (1, 49, 50, 199, 200)], ["small", "small", "medium", "medium", "large"])
        with self.assertRaises(ValueError):
            band_for(0)

    def test_selects_per_band_and_is_deterministic(self):
        stats = make_stats()
        a, short = select_sample(stats, ["Cat"], PER_BAND, 5, 1)
        self.assertEqual(Counter(band_for(s.review_count) for s in a), PER_BAND)
        self.assertEqual(short, [])
        self.assertEqual(a, select_sample(stats, ["Cat"], PER_BAND, 5, 1)[0])
        self.assertNotEqual(a, select_sample(stats, ["Cat"], PER_BAND, 5, 2)[0])

    def test_shortfall_carries_to_next_band(self):
        a, short = select_sample(make_stats(large=3), ["Cat"], PER_BAND, 5, 1)
        counts = Counter(band_for(s.review_count) for s in a)
        self.assertEqual((counts["large"], counts["medium"], counts["small"]), (3, 17, 5))
        self.assertEqual(short, [{"category": "Cat", "band": "large", "requested": 10, "selected": 3}])

    def test_filters_category_min_questions_and_empty_products(self):
        stats = make_stats(large=2, medium=0, small=0) + [
            ProductStats("other", "Other", 300, 6),
            ProductStats("few-questions", "Cat", 300, 4),
            ProductStats("no-reviews", "Cat", 0, 6),
        ]
        a, _ = select_sample(stats, ["Cat"], PER_BAND, 5, 1)
        self.assertEqual(sorted(s.product_id for s in a), ["Cat-large-00", "Cat-large-01"])

    def test_assign_splits_stratified(self):
        selected = make_stats(large=10, medium=0, small=5)
        splits = assign_splits(selected, 0.3, 1)
        dev = Counter(band_for(s.review_count) for s in selected if splits[s.product_id] == "dev")
        self.assertEqual((dev["large"], dev["small"]), (3, 2))
        self.assertEqual(set(splits.values()), {"dev", "test"})
        self.assertEqual(splits, assign_splits(selected, 0.3, 1))

    def test_scan_takes_max_review_count_over_duplicate_records_and_counts_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            products = write_jsonl(tmp / "products.jsonl", [
                {"asin": "A", "category": "Cat", "questions": [], "reviews": [{"reviewText": t} for t in ("a", "b", " ")]},
                {"asin": "A", "category": "Cat", "questions": [], "reviews": [{"reviewText": t} for t in "abcde"]},
                {"asin": "B", "category": "Cat", "questions": [], "reviews": [{"reviewText": "only in products"}]},
            ])
            questions = write_jsonl(tmp / "questions.jsonl", [{"qid": i, "asin": "A"} for i in range(6)])
            stats = scan_product_stats(products, questions)
        self.assertEqual(stats, [ProductStats("A", "Cat", 5, 6)])


if __name__ == "__main__":
    unittest.main()
