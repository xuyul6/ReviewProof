import unittest

from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval.bm25 import BM25Cache, load_product_chunks, tokenize
from tests.support import DatabaseTestCase, make_manifest, make_product


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Fits the 4th-gen iPad!"), ["fits", "the", "4th", "gen", "ipad"])

    def test_chinese_query_has_no_tokens(self):
        self.assertEqual(tokenize("声音大吗？"), [])


class BM25CacheTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        self.cache = BM25Cache()

    def test_ranks_matching_chunk_first_within_product(self):
        hits = self.cache.search(self.conn, "A", "is it loud", top_k=50)
        self.assertEqual([h.chunk.chunk_id for h in hits], ["A-r00001-c00"])

    def test_identical_text_in_other_product_never_returned(self):
        hits = self.cache.search(self.conn, "A", "too loud", top_k=50)
        self.assertTrue(hits)
        self.assertTrue(all(h.chunk.product_id == "A" for h in hits))

    def test_no_shared_terms_means_no_hits(self):
        self.assertEqual(self.cache.search(self.conn, "A", "waterproof", top_k=50), [])
        self.assertEqual(self.cache.search(self.conn, "A", "声音大吗？", top_k=50), [])

    def test_top_k_limits_hits(self):
        self.assertEqual(len(self.cache.search(self.conn, "A", "battery loud ipad", top_k=2)), 2)

    def test_reimport_with_changed_text_rebuilds_index(self):
        self.cache.search(self.conn, "A", "loud", top_k=50)
        load_dataset(self.conn, make_manifest("A"), {"A": make_product("A", ["Quiet and waterproof."])}, {})
        self.assertEqual([h.chunk.chunk_id for h in self.cache.search(self.conn, "A", "waterproof", top_k=50)],
                         ["A-r00000-c00"])

    def test_unknown_product_returns_no_hits(self):
        self.assertEqual(self.cache.search(self.conn, "ZZZ", "loud", top_k=50), [])

    def test_load_product_chunks_in_chunk_id_order(self):
        ids = [c.chunk_id for c in load_product_chunks(self.conn, "A")]
        self.assertEqual(ids, ["A-r00000-c00", "A-r00001-c00", "A-r00002-c00"])


if __name__ == "__main__":
    unittest.main()
