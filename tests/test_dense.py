import unittest

from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval.dense import IncompleteEmbeddingsError, dense_search
from tests.support import DatabaseTestCase, FakeEmbedder, make_manifest, make_product


class DenseSearchTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
        self.vec = FakeEmbedder().embed_query("Too loud for me.")

    def test_exact_text_ranks_first_with_cosine_one(self):
        hits = dense_search(self.conn, "A", self.vec, "fake", top_k=50)
        self.assertEqual(hits[0].chunk.chunk_id, "A-r00001-c00")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        self.assertEqual(len(hits), 3)

    def test_identical_text_in_other_product_never_returned(self):
        hits = dense_search(self.conn, "A", self.vec, "fake", top_k=50)
        self.assertTrue(all(h.chunk.product_id == "A" for h in hits))

    def test_top_k(self):
        self.assertEqual(len(dense_search(self.conn, "A", self.vec, "fake", top_k=2)), 2)

    def test_missing_vector_raises_instead_of_partial_search(self):
        self.conn.execute("UPDATE chunks SET embedding = NULL WHERE chunk_id = 'A-r00002-c00'")
        self.conn.commit()
        with self.assertRaises(IncompleteEmbeddingsError) as ctx:
            dense_search(self.conn, "A", self.vec, "fake", top_k=50)
        self.assertEqual(ctx.exception.missing, 1)

    def test_vectors_from_another_model_raise(self):
        with self.assertRaises(IncompleteEmbeddingsError):
            dense_search(self.conn, "A", self.vec, "other-model", top_k=50)

    def test_no_vector_index_exists(self):
        rows = self.conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'").fetchall()
        self.assertFalse(any("hnsw" in r[0] or "ivfflat" in r[0] for r in rows))


if __name__ == "__main__":
    unittest.main()
