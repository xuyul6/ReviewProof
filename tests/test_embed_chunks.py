import unittest

from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, FakeEmbedder, make_manifest, make_product


class NaNEmbedder(FakeEmbedder):
    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        vectors[-1][0] = float("nan")
        return vectors


class EmbedChunksTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        products = {"A": make_product("A"), "B": make_product("B", ["Cheap plastic.", "Arrived broken."])}
        load_dataset(self.conn, make_manifest("A", "B"), products, {})  # 3 + 2 = 5 chunks

    def count_embedded(self):
        return self.conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]

    def test_embeds_pending_then_nothing_left(self):
        self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 5)
        self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 0)
        self.assertEqual(
            self.conn.execute("SELECT DISTINCT embedding_model FROM chunks").fetchall(), [("fake",)])

    def test_resumes_after_failure(self):
        with self.assertRaises(RuntimeError):
            embed_pending_chunks(self.conn, FakeEmbedder(fail_on_call=2), batch_size=2)
        self.assertEqual(self.count_embedded(), 2)  # the first batch was committed
        self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=2), 3)

    def test_wrong_dimension_rejected_without_writing(self):
        with self.assertRaises(ValueError):
            embed_pending_chunks(self.conn, FakeEmbedder(dim=8), batch_size=2)
        self.assertEqual(self.count_embedded(), 0)

    def test_non_finite_vector_rejected_without_writing_that_batch(self):
        with self.assertRaises(ValueError):
            embed_pending_chunks(self.conn, NaNEmbedder(), batch_size=5)
        self.assertEqual(self.count_embedded(), 0)

    def test_model_change_reembeds_everything(self):
        embed_pending_chunks(self.conn, FakeEmbedder("a"), 2)
        self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder("b"), 2), 5)

    def test_reports_progress_after_each_committed_batch(self):
        seen = []
        embed_pending_chunks(self.conn, FakeEmbedder(), 2, on_batch=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(2, 5), (4, 5), (5, 5)])

    def test_product_filter(self):
        self.assertEqual(embed_pending_chunks(self.conn, FakeEmbedder(), 2, product_ids=["B"]), 2)
        rows = self.conn.execute(
            "SELECT product_id, count(*) FROM chunks WHERE embedding IS NOT NULL GROUP BY product_id").fetchall()
        self.assertEqual(rows, [("B", 2)])


if __name__ == "__main__":
    unittest.main()
