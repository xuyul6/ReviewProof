import unittest

from reviewproof.eval.dataset import EvalQuestion
from reviewproof.eval.retrieval_eval import VARIANTS, evaluate_retrieval, rank_variants
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow.state import RetrievalDeps
from tests.support import DatabaseTestCase, FakeEmbedder, FakeReranker, make_manifest, make_product

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)]
BATTERY = [f"Battery review {i}: the battery lasts two days." for i in range(6)]
EXACT = "Battery review 3: the battery lasts two days."


def question(qid, product="C", strategy="hybrid", answerable=True, text=EXACT):
    return EvalQuestion(qid=qid, product_id=product, category="Electronics", split="dev", strategy=strategy,
                        question_text=text, question_type="descriptive", answerable_silver=answerable,
                        snippets=())


class RetrievalEvalTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "C"), {"A": make_product("A"), "C": make_product("C", NOISE + BATTERY)}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self.deps = RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(),
                                  bm25_cache=BM25Cache(), params=RetrievalParams())

    def test_variants_rank_relevant_chunk(self):
        ranked = rank_variants(self.conn, "C", EXACT, self.deps)
        self.assertEqual(set(ranked), set(VARIANTS))
        self.assertTrue(all(ranked[v] for v in VARIANTS))
        self.assertEqual(ranked["hybrid_rerank"][0], "C-r00009-c00")
        self.assertEqual(ranked["dense"][0], "C-r00009-c00")
        self.assertEqual(set(ranked["hybrid"]), set(ranked["hybrid_rerank"]))
        self.assertTrue(all(cid.startswith("C-") for v in VARIANTS for cid in ranked[v]))

    def test_exclusions_and_scores(self):
        questions = [question("q_fc", product="A", strategy="full_context"), question("q_no", answerable=False),
                     question("q_none"), question("q_ok")]
        qrels = {"q_fc": frozenset({"A-r00001-c00"}), "q_no": frozenset({"C-r00009-c00"}),
                 "q_none": frozenset(), "q_ok": frozenset({"C-r00009-c00"})}
        rows = {r["qid"]: r for r in evaluate_retrieval(self.conn, questions, qrels, self.deps)}
        self.assertEqual({q: rows[q]["exclude_reason"] for q in rows},
                         {"q_fc": "full_context", "q_no": "silver_unanswerable", "q_none": "no_aligned_snippet",
                          "q_ok": None})
        self.assertFalse(any(rows[q]["included"] for q in ("q_fc", "q_no", "q_none")))
        ok = rows["q_ok"]
        self.assertTrue(ok["included"])
        self.assertEqual(ok["n_relevant"], 1)
        self.assertEqual(ok["hybrid_rerank"]["hit@5"], 1.0)
        self.assertEqual(set(ok["dense"]), {"recall@5", "recall@10", "recall@20", "recall@40", "hit@5", "mrr@10", "ndcg@5"})
        self.assertGreaterEqual(ok["latency_ms"], 0)
        self.assertNotIn("bm25", rows["q_fc"])

    def test_channel_error_becomes_null(self):
        deps = RetrievalDeps(embedder=FakeEmbedder(fail_queries=True), embedding_model="fake", reranker=FakeReranker(),
                             bm25_cache=BM25Cache(), params=RetrievalParams())
        [row] = evaluate_retrieval(self.conn, [question("q_ok")], {"q_ok": frozenset({"C-r00009-c00"})}, deps)
        self.assertIsNone(row["dense"])
        self.assertIsNotNone(row["bm25"])
        self.assertTrue(row["included"])


if __name__ == "__main__":
    unittest.main()
