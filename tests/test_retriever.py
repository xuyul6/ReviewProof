import unittest

from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval import ProductNotFoundError, RetrievalParams, retrieve
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.retriever import rrf_fuse
from tests.support import DatabaseTestCase, FakeEmbedder, FakeReranker, make_manifest, make_product


class RRFTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        fused = rrf_fuse({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=60)
        self.assertEqual([d for d, _ in fused], ["b", "a", "c"])
        self.assertAlmostEqual(dict(fused)["b"], 1 / 62 + 1 / 61)

    def test_ties_break_by_id(self):
        self.assertEqual([d for d, _ in rrf_fuse({"dense": ["z"], "bm25": ["a"]})], ["a", "z"])


class RetrieveTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
        self.kw = dict(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(), bm25_cache=BM25Cache())

    def run_retrieve(self, query="Too loud for me.", strategy="hybrid", **overrides):
        return retrieve(self.conn, "A", query, strategy=strategy, **{**self.kw, **overrides})

    def status(self, result, name):
        return next(c for c in result.channels if c.name == name)

    def test_small_product_uses_full_context_on_auto(self):
        r = self.run_retrieve(strategy="auto")
        self.assertEqual((r.strategy, r.mode, r.service_status), ("full_context", "full_context", "ok"))
        self.assertEqual([c.chunk.chunk_id for c in r.candidates], ["A-r00000-c00", "A-r00001-c00", "A-r00002-c00"])
        self.assertEqual(r.pool, r.candidates)
        self.assertEqual({(c.status, c.reason) for c in r.channels}, {("skipped", "full_context")})

    def test_threshold_zero_forces_hybrid_on_auto(self):
        r = self.run_retrieve(strategy="auto", params=RetrievalParams(full_context_max_tokens=0))
        self.assertEqual(r.strategy, "hybrid")

    def test_hybrid_all_channels_ok(self):
        r = self.run_retrieve()
        self.assertEqual((r.mode, r.service_status), ("dense+bm25+rerank", "ok"))
        top = r.candidates[0]
        self.assertEqual(top.chunk.chunk_id, "A-r00001-c00")
        self.assertEqual((top.dense_rank, top.bm25_rank), (1, 1))
        self.assertIsNotNone(top.rerank_score)
        self.assertTrue(all(c.latency_ms >= 0 for c in r.channels))

    def test_fewer_chunks_than_k_returns_each_chunk_once(self):
        r = self.run_retrieve(query="loud battery ipad")
        ids = [c.chunk.chunk_id for c in r.candidates]
        self.assertLessEqual(len(ids), 3)
        self.assertEqual(len(ids), len(set(ids)))

    def test_evidence_k_caps_candidates(self):
        r = self.run_retrieve(params=RetrievalParams(evidence_k=1))
        self.assertEqual(len(r.candidates), 1)
        self.assertEqual(len(r.pool), 3)

    def test_only_requested_product(self):
        r = self.run_retrieve()
        self.assertTrue(all(c.chunk.product_id == "A" for c in r.pool + r.candidates))

    def test_dense_failure_degrades_to_bm25(self):
        r = self.run_retrieve(embedder=FakeEmbedder(fail_queries=True))
        self.assertEqual((r.mode, r.service_status), ("bm25+rerank", "degraded"))
        self.assertEqual(self.status(r, "dense").status, "failed")
        self.assertTrue(r.candidates)

    def test_incomplete_embeddings_reported_as_dense_failure(self):
        r = self.run_retrieve(embedding_model="other-model")
        self.assertEqual(self.status(r, "dense").reason.split(":")[0], "IncompleteEmbeddingsError")
        self.assertEqual(self.status(r, "bm25").status, "ok")

    def test_dense_sql_error_does_not_poison_bm25(self):
        # An 8-dim query vector makes pgvector raise inside Postgres, aborting the transaction.
        r = self.run_retrieve(embedder=FakeEmbedder(dim=8))
        self.assertEqual(self.status(r, "dense").status, "failed")
        self.assertEqual(self.status(r, "bm25").status, "ok")
        self.assertEqual(r.candidates[0].chunk.chunk_id, "A-r00001-c00")

    def test_chinese_query_skips_bm25_without_degrading(self):
        r = self.run_retrieve(query="声音大吗？")
        self.assertEqual((self.status(r, "bm25").status, self.status(r, "bm25").reason), ("skipped", "no_query_terms"))
        self.assertEqual((r.mode, r.service_status), ("dense+rerank", "ok"))

    def test_both_channels_failed_is_service_failure_not_empty_answer(self):
        r = self.run_retrieve(query="声音大吗？", embedder=FakeEmbedder(fail_queries=True))
        self.assertEqual((r.mode, r.service_status, r.candidates), ("none", "failed", ()))

    def test_failed_channel_and_empty_pool_is_service_failure(self):
        # BM25 runs fine but finds nothing; with dense down, that emptiness comes from the outage.
        r = self.run_retrieve(query="waterproof", embedder=FakeEmbedder(fail_queries=True))
        self.assertEqual((self.status(r, "bm25").status, self.status(r, "bm25").hits), ("ok", 0))
        self.assertEqual((r.mode, r.service_status, r.candidates), ("none", "failed", ()))

    def test_reranker_failure_falls_back_to_rrf_order(self):
        r = self.run_retrieve(reranker=FakeReranker(fail=True))
        self.assertEqual((r.mode, r.service_status), ("dense+bm25", "degraded"))
        self.assertEqual([c.chunk.chunk_id for c in r.candidates], [c.chunk.chunk_id for c in r.pool][:8])

    def test_reranker_disabled_is_not_degraded(self):
        r = self.run_retrieve(reranker=None)
        self.assertEqual((r.mode, r.service_status), ("dense+bm25", "ok"))
        self.assertEqual((self.status(r, "rerank").status, self.status(r, "rerank").reason), ("skipped", "disabled"))

    def test_unknown_product_and_bad_input(self):
        with self.assertRaises(ProductNotFoundError):
            retrieve(self.conn, "ZZZ", "loud", **self.kw)
        with self.assertRaises(ValueError):
            self.run_retrieve(query="   ")
        with self.assertRaises(ValueError):
            self.run_retrieve(strategy="dense")


if __name__ == "__main__":
    unittest.main()
