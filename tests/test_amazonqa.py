import hashlib
import tempfile
import unittest
from pathlib import Path

from reviewproof.ingest.amazonqa import file_sha256, load_products, load_questions
from tests.support import write_jsonl


def product(asin, reviews, category="Electronics"):
    return {
        "asin": asin,
        "category": category,
        "questions": [],
        "reviews": [r if isinstance(r, dict) else {"reviewText": r, "helpful": [0, 0]} for r in reviews],
    }


def question(qid, asin, text, answerable):
    return {
        "qid": qid, "asin": asin, "category": "Electronics", "questionText": text,
        "questionType": "yesno", "is_answerable": answerable,
        "answers": [{"answerText": "yes", "answerType": "NA", "helpful": [0, 0]}],
        "review_snippets": ["snippet"],
    }


class AmazonQAReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rows):
        return write_jsonl(self.dir / "data.jsonl", rows)

    def test_merges_duplicate_records_without_duplicate_reviews(self):
        path = self.write([
            product("A", ["Great", "Bad"]),
            product("B", ["x"]),
            product("A", ["great ", "New one"]),
        ])
        p = load_products(path, {"A"})["A"]
        self.assertEqual([r.text for r in p.reviews], ["Great", "Bad", "New one"])
        self.assertEqual([r.review_id for r in p.reviews], ["A-r00000", "A-r00001", "A-r00002"])
        self.assertEqual((p.reviews[2].source_line, p.reviews[2].source_position), (3, 1))

    def test_keeps_duplicate_reviews_within_one_record(self):
        path = self.write([product("A", ["Works", "works"])])
        self.assertEqual(len(load_products(path, {"A"})["A"].reviews), 2)

    def test_skips_empty_reviews_without_consuming_index(self):
        path = self.write([product("A", ["", "   ", "Real"])])
        r = load_products(path, {"A"})["A"].reviews
        self.assertEqual([(x.review_index, x.source_position, x.text) for x in r], [(0, 2, "Real")])

    def test_keeps_raw_text_and_stores_unescaped_text(self):
        path = self.write([product("A", ["It&#39;s  fine"])])
        r = load_products(path, {"A"})["A"].reviews[0]
        self.assertEqual((r.raw_text, r.text), ("It&#39;s  fine", "It's fine"))

    def test_bad_helpful_defaults_to_zero(self):
        path = self.write([product("A", [
            {"reviewText": "one"},
            {"reviewText": "two", "helpful": [1]},
            {"reviewText": "three", "helpful": [2, 5]},
        ])])
        reviews = load_products(path, {"A"})["A"].reviews
        self.assertEqual([(x.helpful_yes, x.helpful_total) for x in reviews], [(0, 0), (0, 0), (2, 5)])

    def test_only_requested_products_are_loaded(self):
        path = self.write([product("A", ["a"]), product("B", ["b"])])
        self.assertEqual(set(load_products(path, {"A", "missing"})), {"A"})

    def test_load_questions_keeps_labels_and_duplicate_texts(self):
        path = self.write([
            question(7, "A", "Is it loud?", 1),
            question(8, "A", "Is it loud?", 0),
            question(9, "B", "Other?", 1),
        ])
        qs = load_questions(path, {"A"})
        self.assertEqual([(q.qid, q.is_answerable) for q in qs["A"]], [("7", True), ("8", False)])
        self.assertNotIn("B", qs)
        self.assertEqual(qs["A"][0].review_snippets, ("snippet",))

    def test_file_sha256_matches_hashlib(self):
        path = self.write([product("A", ["a"])])
        self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
