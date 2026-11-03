import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewproof.eval import workflow_eval
from reviewproof.eval.baselines import B0Answer
from reviewproof.eval.dataset import EvalQuestion
from reviewproof.eval.workflow_eval import cost_usd, run_workflow_eval
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.llm import LLMError
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow.schemas import Claim, EvidenceAssessment, GeneratedAnswer, QueryPlan
from reviewproof.workflow.state import RetrievalDeps
from tests.support import DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest, make_product

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)]
SECRET_Q = Question(qid="qa", product_id="A", question_text="Is it loud?", question_type="yesno", is_answerable=True,
                    answers=({"answerText": "SECRET-ANSWER"},), review_snippets=("SECRET-SNIPPET",))


def first_chunk(prompt):
    return re.search(r"\[([A-Z]+-r\d{5}-c\d{2})\]", prompt).group(1)


def plan(prompt):
    return QueryPlan(route="review_question", out_of_scope_reason="", is_yes_no=True, search_query="loud")


def assess(prompt):
    return EvidenceAssessment(status="answered", supporting_chunk_ids=[first_chunk(prompt)], missing=[],
                              rewrite_query="")


def generate(prompt):
    return GeneratedAnswer(claims=[Claim(text="It is loud.", chunk_ids=[first_chunk(prompt)])], unknowns=[],
                           verdict="yes")


def b0(prompt):
    return B0Answer(answered=True, answer="Probably quiet.")


def script(n=10, **overrides):
    s = {"QueryPlan": [plan] * n, "EvidenceAssessment": [assess] * n, "GeneratedAnswer": [generate] * n,
         "B0Answer": [b0] * n}
    s.update(overrides)
    return s


def eq(qid, product, strategy):
    return EvalQuestion(qid=qid, product_id=product, category="Electronics", split="dev", strategy=strategy,
                        question_text="Is it loud?", question_type="yesno", answerable_silver=True,
                        snippets=("SECRET-SNIPPET",))


QUESTIONS = [eq("qa", "A", "full_context"), eq("qc", "C", "hybrid")]


class WorkflowEvalTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "C"), {"A": make_product("A"), "C": make_product("C", NOISE)},
                     {"A": [SECRET_Q]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self.deps = RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(),
                                  bm25_cache=BM25Cache(), params=RetrievalParams(full_context_max_tokens=10))
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "per_question.jsonl"

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def run_eval(self, llm, systems=workflow_eval.SYSTEMS, questions=QUESTIONS):
        self.llm = llm
        return run_workflow_eval(self.conn, questions, llm=llm, deps=self.deps, max_rewrites=1, systems=systems,
                                 out_path=self.out)

    def file_rows(self):
        return [json.loads(line) for line in self.out.read_text().splitlines()]

    def test_rows_written_per_system(self):
        rows = self.run_eval(FakeLLM(script()))
        self.assertEqual(sorted((r["qid"], r["system"]) for r in rows),
                         [("qa", "b0_no_evidence"), ("qa", "workflow"), ("qc", "b0_no_evidence"),
                          ("qc", "b1_full_context"), ("qc", "workflow")])
        self.assertEqual(rows, self.file_rows())
        wf = next(r for r in rows if (r["qid"], r["system"]) == ("qc", "workflow"))
        self.assertEqual((wf["status"], wf["answered_binary"], wf["citation_out_of_product"]), ("answered", True, 0))
        self.assertEqual(wf["retrieval_modes"], ["dense+bm25+rerank"])
        b1 = next(r for r in rows if r["system"] == "b1_full_context")
        self.assertEqual(b1["retrieval_modes"], ["full_context"])
        b0row = next(r for r in rows if (r["qid"], r["system"]) == ("qa", "b0_no_evidence"))
        self.assertEqual((b0row["answered_binary"], b0row["llm_calls"]), (True, 1))
        for key in ("cost_usd", "latency_ms", "input_tokens", "answer_text", "error", "service_status"):
            self.assertIn(key, wf)

    def test_resume_skips_done_and_drops_partial_line(self):
        done = {"qid": "qa", "system": "workflow", "status": "answered", "marker": "original"}
        self.out.write_text(json.dumps(done) + "\n" + '{"qid": "qc", "sys', encoding="utf-8")
        self.run_eval(FakeLLM(script(n=1)), systems=("workflow",))
        rows = self.file_rows()
        self.assertEqual([(r["qid"], r["system"]) for r in rows], [("qa", "workflow"), ("qc", "workflow")])
        self.assertEqual(rows[0]["marker"], "original")
        self.assertIsNone(rows[1]["error"])

    def test_resume_retries_service_failures(self):
        self.run_eval(FakeLLM(script(EvidenceAssessment=[LLMError("rate limited")])), systems=("workflow",),
                      questions=[QUESTIONS[0]])
        self.assertEqual(self.file_rows()[0]["service_status"], "failed")
        self.run_eval(FakeLLM(script(n=1)), systems=("workflow",), questions=[QUESTIONS[0]])
        rows = self.file_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["service_status"], rows[0]["status"]), ("ok", "answered"))

    def test_workflow_rows_never_call_the_note(self):
        unanswerable = EvidenceAssessment(status="unanswerable", supporting_chunk_ids=[], missing=[], rewrite_query="")
        rows = self.run_eval(FakeLLM(script(EvidenceAssessment=[unanswerable] * 10)), systems=("workflow",))
        self.assertEqual([(r["status"], r.get("error")) for r in rows], [("unanswerable", None)] * 2)
        self.assertNotIn("GeneralNote", [c[0] for c in self.llm.calls])

    def test_service_failure_row(self):
        rows = self.run_eval(FakeLLM(script(EvidenceAssessment=[LLMError("down")])), systems=("workflow",),
                             questions=[QUESTIONS[0]])
        self.assertEqual((rows[0]["service_status"], rows[0]["answered_binary"], rows[0]["status"]),
                         ("failed", None, None))

    def test_unexpected_exception_is_recorded_and_run_continues(self):
        real = workflow_eval.answer_question
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        with patch.object(workflow_eval, "answer_question", side_effect=flaky):
            rows = self.run_eval(FakeLLM(script()), systems=("workflow",))
        self.assertEqual(rows[0]["error"], "RuntimeError: boom")
        self.assertEqual((rows[0]["service_status"], rows[0]["answered_binary"]), ("failed", None))
        self.assertEqual(rows[1]["status"], "answered")

    def test_no_snippet_or_answer_leaks_into_prompts(self):
        self.run_eval(FakeLLM(script()))
        self.assertTrue(self.llm.calls)
        for _, system, prompt in self.llm.calls:
            for secret in ("SECRET-SNIPPET", "SECRET-ANSWER"):
                self.assertNotIn(secret, system + prompt)

    def test_cost_prices_each_call_at_its_own_tier(self):
        self.assertAlmostEqual(cost_usd([(100_000, 0)] * 10), 0.10)
        self.assertAlmostEqual(cost_usd([(200_000, 1_000_000)]), 0.10 + 2.50)
        # a short plan call next to a long assess call: the long call is still long-tier
        self.assertAlmostEqual(cost_usd([(500, 0), (150_000, 0)]), (500 * 0.10 + 150_000 * 0.50) / 1_000_000)
        self.assertEqual(cost_usd([]), 0.0)

    def test_row_cost_comes_from_per_call_usage(self):
        rows = self.run_eval(FakeLLM(script()), systems=("workflow",), questions=[QUESTIONS[0]])
        self.assertEqual(len(rows[0]["usage"]), rows[0]["llm_calls"])
        self.assertAlmostEqual(rows[0]["cost_usd"], cost_usd([tuple(u) for u in rows[0]["usage"]]))


if __name__ == "__main__":
    unittest.main()
