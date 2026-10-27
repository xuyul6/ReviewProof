import unittest

from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.llm import LLMError
from reviewproof.runs import get_run, new_run_id, save_run
from reviewproof.workflow import RetrievalDeps, answer_question
from reviewproof.workflow.schemas import Claim, EvidenceAssessment, GeneratedAnswer, QueryPlan
from tests.support import DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest, make_product


def answered_script(assessment=None):
    return {"QueryPlan": [QueryPlan(route="review_question", out_of_scope_reason="", is_yes_no=True,
                                    search_query="loud")],
            "EvidenceAssessment": [assessment or EvidenceAssessment(
                status="answered", supporting_chunk_ids=["A-r00001-c00"], missing=[], rewrite_query="")],
            "GeneratedAnswer": [GeneratedAnswer(claims=[Claim(text="It is too loud.", chunk_ids=["A-r00001-c00"])],
                                                unknowns=[], verdict="yes")]}


class RunStoreTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A"), {"A": make_product("A")}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
        self.deps = RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(),
                                  bm25_cache=BM25Cache(), params=RetrievalParams(evidence_k=2, full_context_max_tokens=0))

    def run_workflow(self, script):
        return answer_question(self.conn, "A", "Is it loud?", llm=FakeLLM(script), retrieval=self.deps)

    def test_save_and_get_round_trip(self):
        result = self.run_workflow(answered_script())
        run_id = new_run_id()
        save_run(self.conn, run_id, result, "fake-llm")
        run = get_run(self.conn, run_id)
        self.assertEqual((run["run_id"], run["llm_model"], run["status"]), (run_id, "fake-llm", "answered"))
        self.assertEqual(run["answer_text"], result.answer_text)
        self.assertEqual(run["citations"][0]["chunk_id"], "A-r00001-c00")
        self.assertEqual([t["node"] for t in run["trace"]], [t.node for t in result.trace])
        self.assertIsInstance(run["created_at"], str)

    def test_snapshots_round_trip(self):
        result = self.run_workflow(answered_script())
        run_id = new_run_id()
        save_run(self.conn, run_id, result, "fake-llm")
        run = get_run(self.conn, run_id)
        self.assertTrue(run["retrievals"][0]["candidates"])
        self.assertEqual(run["retrievals"][0]["query"], "loud")

    def test_get_unknown_returns_none(self):
        self.assertIsNone(get_run(self.conn, "nope"))

    def test_failed_run_is_saved(self):
        script = answered_script()
        script["EvidenceAssessment"] = [LLMError("down")]
        result = self.run_workflow(script)
        run_id = new_run_id()
        save_run(self.conn, run_id, result, "fake-llm")
        run = get_run(self.conn, run_id)
        self.assertEqual((run["status"], run["service_status"]), (None, "failed"))

    def test_run_ids_are_unique_hex(self):
        ids = {new_run_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)
        self.assertTrue(all(len(i) == 32 and int(i, 16) >= 0 for i in ids))


if __name__ == "__main__":
    unittest.main()
