import unittest

import numpy as np

from reviewproof.db import EMBEDDING_DIM
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import LoadReport, load_dataset
from tests.support import REVIEW_TEXTS, DatabaseTestCase, make_manifest, make_product

QUESTIONS = {"A": [Question(
    qid="7", product_id="A", question_text="QUESTION_SENTINEL_42 is it loud?", question_type="yesno",
    is_answerable=True, answers=({"answerText": "ANSWER_SENTINEL_42 yes", "answerType": "NA", "helpful": [0, 0]},),
    review_snippets=("Too loud for me.",),
)]}


class LoaderTests(DatabaseTestCase):
    def load(self, product=None, manifest=None, questions=QUESTIONS):
        product = product or make_product()
        return load_dataset(self.conn, manifest or make_manifest("A"), {"A": product}, questions)

    def scalar(self, sql, params=None):
        # Passing params (even empty) makes psycopg parse '%' in LIKE patterns as placeholders.
        return self.conn.execute(sql, params).fetchone()[0]

    def test_load_is_idempotent(self):
        first = self.load()
        second = self.load()
        self.assertEqual(first, LoadReport(products=1, reviews=3, chunks=3, questions=1))
        self.assertEqual(first, second)
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks"), first.chunks)

    def test_embedding_kept_when_text_unchanged_and_cleared_when_changed(self):
        self.load()
        self.conn.execute("UPDATE chunks SET embedding = %s, embedding_model = 'm'",
                          (np.ones(EMBEDDING_DIM, dtype=np.float32),))
        self.conn.commit()
        self.load()
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks WHERE embedding_model = 'm'"), 3)

        edited = make_product(texts=[REVIEW_TEXTS[0], "Actually it is quiet.", REVIEW_TEXTS[2]])
        self.load(product=edited)
        rows = dict(self.conn.execute(
            "SELECT review_id, embedding IS NULL AND embedding_model IS NULL FROM chunks").fetchall())
        self.assertEqual(rows, {"A-r00000": False, "A-r00001": True, "A-r00002": False})

    def test_stale_rows_are_removed(self):
        self.load()
        report = self.load(product=make_product(texts=REVIEW_TEXTS[:2]), questions={})
        self.assertEqual(report, LoadReport(products=1, reviews=2, chunks=2, questions=0))
        self.assertEqual(self.scalar("SELECT review_count FROM products WHERE product_id = 'A'"), 2)

    def test_missing_manifest_product_raises_and_writes_nothing(self):
        with self.assertRaisesRegex(ValueError, "B"):
            self.load(manifest=make_manifest("A", "B"))
        self.assertEqual(self.scalar("SELECT count(*) FROM datasets"), 0)
        self.assertEqual(self.scalar("SELECT count(*) FROM products"), 0)

    def test_question_fields_never_reach_chunks(self):
        self.load()
        self.assertEqual(self.scalar("SELECT count(*) FROM chunks WHERE text LIKE '%SENTINEL_42%'"), 0)
        self.assertEqual(self.scalar("SELECT count(*) FROM reviews WHERE text LIKE '%SENTINEL_42%'"), 0)
        row = self.conn.execute(
            "SELECT is_answerable_silver, answers->0->>'answerText', review_snippets->>0 FROM questions WHERE qid = '7'"
        ).fetchone()
        self.assertEqual(row, (True, "ANSWER_SENTINEL_42 yes", "Too loud for me."))

    def test_product_split_band_and_counts_stored(self):
        self.load()
        row = self.conn.execute(
            "SELECT split, band, review_count, chunk_count, token_estimate FROM products WHERE product_id = 'A'").fetchone()
        expected_tokens = sum(-(-len(t) // 4) for t in REVIEW_TEXTS)
        self.assertEqual(row, ("dev", "small", 3, 3, expected_tokens))
        self.assertEqual(self.scalar("SELECT source_sha256 FROM datasets WHERE dataset_id = 'test-dataset'"), "abc")


if __name__ == "__main__":
    unittest.main()
