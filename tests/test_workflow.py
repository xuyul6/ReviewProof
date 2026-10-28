import re
import unittest
from unittest.mock import patch

import psycopg

from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.llm import LLMError
from reviewproof.retrieval import ProductNotFoundError, RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow import QAResult, RetrievalDeps, answer_question, stream_answer
from reviewproof.workflow.state import TraceEntry
from reviewproof.workflow import nodes as workflow_nodes
from reviewproof.workflow.schemas import (Claim, EvidenceAssessment, GeneratedAnswer, GeneratedYesNoAnswer,
                                          QueryPlan)
from tests.support import DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest, make_product

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)]
BATTERY = [f"Battery review {i}: the battery lasts two days." for i in range(6)]
SECRET_Q = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno",
                    is_answerable=True, answers=({"answerText": "SECRET-ANSWER"},),
                    review_snippets=("SECRET-SNIPPET",))


def plan(query, yes_no=True, route="review_question", reason=""):
    return QueryPlan(route=route, out_of_scope_reason=reason, is_yes_no=yes_no, search_query=query)


def assess(status, ids=(), missing=(), rewrite=""):
    return EvidenceAssessment(status=status, supporting_chunk_ids=list(ids), missing=list(missing),
                              rewrite_query=rewrite)


def generated(*claims, unknowns=(), verdict="yes"):
    return GeneratedAnswer(claims=[Claim(text=t, chunk_ids=list(ids)) for t, ids in claims],
                           unknowns=list(unknowns), verdict=verdict)


LOUD = ("A buyer says it is too loud.", ["A-r00001-c00"])


def cite_first_candidate(prompt):
    first = re.search(r"\[([A-Z]+-r\d{5}-c\d{2})\]", prompt).group(1)
    return generated(("Cited from the excerpts.", [first]))


class WorkflowTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B", "C"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."]),
                      "C": make_product("C", NOISE + BATTERY)},
                     {"A": [SECRET_Q]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self.deps = self.make_deps(RetrievalParams(evidence_k=2, full_context_max_tokens=0))

    def make_deps(self, params=RetrievalParams(), embedder=None):
        return RetrievalDeps(embedder=embedder or FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(),
                             bm25_cache=BM25Cache(), params=params)

    def ask(self, script, question="Is it loud?", product="A", deps=None, max_rewrites=1):
        self.llm = FakeLLM(script)
        return answer_question(self.conn, product, question, llm=self.llm, retrieval=deps or self.deps,
                               max_rewrites=max_rewrites)

    def nodes(self, result):
        return [t.node for t in result.trace]

    def calls(self, name):
        return [c for c in self.llm.calls if c[0] == name]

    def test_answered_path_cites_only_candidates(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(("A buyer says it is too loud.", ["A-r00001-c00", "B-x"]))]})
        self.assertEqual((r.status, r.service_status, r.verdict, r.route), ("answered", "ok", "yes", "review_question"))
        self.assertEqual([c.chunk_id for c in r.citations], ["A-r00001-c00"])
        self.assertEqual(self.nodes(r), ["plan", "retrieve", "assess", "generate", "verify"])
        self.assertEqual((r.llm_calls, r.input_tokens, r.output_tokens), (3, 30, 15))
        self.assertEqual(r.search_queries, ("loud",))
        self.assertIn("[1]", r.answer_text)

    def test_unanswerable_skips_generation(self):
        r = self.ask({"QueryPlan": [plan("waterproof")], "EvidenceAssessment": [assess("unanswerable", missing=["x"])]})
        self.assertEqual((r.status, r.verdict, r.unknowns, r.llm_calls), ("unanswerable", "unknown", (), 2))
        self.assertNotIn("generate", self.nodes(r))
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")

    def test_out_of_scope_skips_retrieval(self):
        r = self.ask({"QueryPlan": [plan("price", route="out_of_scope", reason="price is not in reviews")]},
                     question="How much does it cost?")
        self.assertEqual((r.route, r.status, r.verdict, self.nodes(r)), ("out_of_scope", "unanswerable", None, ["plan"]))
        self.assertIn("price is not in reviews", r.answer_text)

    def test_partial_then_rewrite_finds_new_evidence(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", missing=["battery"], rewrite="battery"),
                                             lambda prompt: assess("answered", ["C-r00006-c00"])],
                      "GeneratedAnswer": [generated(("Battery lasts two days.", ["C-r00006-c00"]), verdict="yes")]},
                     product="C", question="Is it quiet and does the battery last?")
        self.assertEqual((r.rewrites_used, r.search_queries, len(r.retrieval_modes)), (1, ("noise", "battery"), 2))
        self.assertEqual(self.nodes(r), ["plan", "retrieve", "assess", "rewrite", "assess", "generate", "verify"])
        second_prompt = self.calls("EvidenceAssessment")[1][2]
        self.assertIn("Battery review", second_prompt)
        self.assertIn("Noise review", second_prompt)  # earlier candidates are kept

    def test_rewrite_stops_without_new_evidence(self):
        deps = self.make_deps(RetrievalParams(evidence_k=3, full_context_max_tokens=0))  # all 3 chunks of A
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["night"], rewrite="too loud night")],
                      "GeneratedAnswer": [generated(LOUD)]}, deps=deps)
        self.assertEqual(len(self.calls("EvidenceAssessment")), 1)
        rewrite = next(t for t in r.trace if t.node == "rewrite")
        self.assertIn("no_new_evidence", rewrite.detail)
        self.assertEqual(r.status, "partial")

    def test_rewrite_bounded_by_max_rewrites(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery"),
                                             assess("partial", ["C-r00000-c00"], ["charging"], rewrite="charging")],
                      "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]}, product="C")
        self.assertEqual(r.rewrites_used, 1)
        self.assertEqual(self.nodes(r).count("rewrite"), 1)

    def test_same_rewrite_query_is_not_reused(self):
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["night"], rewrite="loud")],
                      "GeneratedAnswer": [generated(LOUD)]})
        self.assertEqual(r.rewrites_used, 0)

    def test_max_rewrites_zero_never_rewrites(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery")],
                      "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]},
                     product="C", max_rewrites=0)
        self.assertEqual(r.rewrites_used, 0)

    def test_full_context_product_never_rewrites(self):
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["night"], rewrite="night noise")],
                      "GeneratedAnswer": [generated(LOUD)]}, deps=self.make_deps())
        self.assertEqual((r.rewrites_used, r.retrieval_modes), (0, ("full_context",)))

    def test_yes_no_questions_generate_with_a_required_verdict(self):
        self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["x"])],
                  "GeneratedAnswer": [generated(LOUD)]})
        self.assertEqual([c[0] for c in self.llm.calls], ["QueryPlan", "EvidenceAssessment", "GeneratedYesNoAnswer"])
        self.ask({"QueryPlan": [plan("loud", yes_no=False)], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                  "GeneratedAnswer": [generated(LOUD, verdict=None)]})
        self.assertEqual(self.llm.calls[-1][0], "GeneratedAnswer")

    def test_yes_no_verdict_unknown_caps_the_run_at_partial(self):
        close_case = GeneratedYesNoAnswer(verdict_question="Does it fit a 2015 Wrangler?", claims=[Claim(
            text="One reviewer says it fit their 2014 Wrangler.", chunk_ids=["A-r00001-c00"])], unknowns=[],
            verdict="unknown")
        r = self.ask({"QueryPlan": [plan("2015 wrangler fit")],
                      "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedYesNoAnswer": [close_case]})
        self.assertEqual((r.status, r.verdict, [c.chunk_id for c in r.citations]),
                         ("partial", "unknown", ["A-r00001-c00"]))
        self.assertIn("downgraded_from=answered", r.trace[-1].detail)
        self.assertIn("verdict_question='Does it fit a 2015 Wrangler?'", r.trace[-2].detail)
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nOne reviewer says it fit their 2014 Wrangler. [1]")

    def test_yes_no_verdict_unknown_without_valid_citations_is_unanswerable(self):
        foreign = GeneratedYesNoAnswer(verdict_question="Does it fit a 2015 Wrangler?", claims=[Claim(
            text="One reviewer says it fit their 2014 Wrangler.", chunk_ids=["B-r00000-c00"])], unknowns=[],
            verdict="unknown")
        r = self.ask({"QueryPlan": [plan("2015 wrangler fit")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["2015"])],
                      "GeneratedYesNoAnswer": [foreign]})
        self.assertEqual((r.status, r.verdict, r.citations), ("unanswerable", "unknown", ()))
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")

    def test_plan_failure_degrades_with_heuristic(self):
        r = self.ask({"QueryPlan": [LLMError("boom")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD, verdict="yes")]})
        self.assertEqual((r.service_status, r.status, r.verdict), ("degraded", "answered", "yes"))
        self.assertEqual(r.search_queries, ("Is it loud?",))
        self.assertEqual(r.trace[0].status, "degraded")

    def test_heuristic_yes_no_detection(self):
        for question, expected in (("Does it fit an iPad?", "unknown"), ("声音大吗？", "unknown"),
                                   ("How long does the battery last?", None)):
            r = self.ask({"QueryPlan": [LLMError("boom")], "EvidenceAssessment": [assess("unanswerable")]},
                         question=question)
            self.assertEqual(r.verdict, expected, question)

    def test_assess_failure_is_service_failure(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [LLMError("API down")]})
        self.assertEqual((r.status, r.service_status, r.verdict), (None, "failed", None))
        self.assertIn("service error", r.answer_text)
        self.assertEqual(r.trace[-1].status, "failed")

    def test_generate_failure_is_service_failure(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [LLMError("timeout")]})
        self.assertEqual((r.status, r.service_status), (None, "failed"))
        self.assertNotIn("don't confirm", r.answer_text)

    def test_retrieval_failure_is_service_failure(self):
        deps = self.make_deps(RetrievalParams(evidence_k=2, full_context_max_tokens=0),
                              embedder=FakeEmbedder(fail_queries=True))
        r = self.ask({"QueryPlan": [plan("声音")]}, deps=deps)  # BM25 skipped, dense failed
        self.assertEqual((r.status, r.service_status), (None, "failed"))
        self.assertEqual(self.calls("EvidenceAssessment"), [])

    def test_all_claims_unsupported_downgrades(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(("Made up.", ["nope"]))]})
        self.assertEqual((r.status, r.citations, r.verdict), ("unanswerable", (), "unknown"))
        verify = next(t for t in r.trace if t.node == "verify")
        self.assertIn("nope", verify.detail)

    def test_supporting_ids_outside_candidates_are_ignored(self):
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("answered", ["A-r00001-c00", "B-r00000-c00"])],
                      "GeneratedAnswer": [generated(LOUD)]})
        assess_entry = next(t for t in r.trace if t.node == "assess")
        self.assertIn("B-r00000-c00", assess_entry.detail)

    def test_chinese_question_uses_english_search_and_chinese_copy(self):
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["夜间"])],
                      "GeneratedAnswer": [generated(("有买家说太吵。", ["A-r00001-c00"]), verdict="yes")]},
                     question="声音大吗？")
        self.assertEqual((r.search_queries[0], r.status), ("loud", "partial"))
        self.assertEqual(r.answer_text, "结论：是\n有买家说太吵。 [1]\n评论没有提到：夜间")

    def test_no_leakage_into_any_prompt(self):
        prompts = []
        self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                  "GeneratedAnswer": [generated(LOUD)]})
        prompts += self.llm.calls
        self.ask({"QueryPlan": [plan("price", route="out_of_scope", reason="price")]}, question="How much?")
        prompts += self.llm.calls
        self.assertEqual(len(prompts), 4)
        for _, system, prompt in prompts:
            for secret in ("SECRET-ANSWER", "SECRET-SNIPPET"):
                self.assertNotIn(secret, system + prompt)

    def test_assessor_decides_status_not_generator_caveats(self):
        # Real runs: the generator filled unknowns with caveats and every answered became partial.
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD, unknowns=["only one reviewer mentions it"])]})
        self.assertEqual((r.status, r.unknowns), ("answered", ()))

    def test_partial_unknowns_come_from_assessment(self):
        r = self.ask({"QueryPlan": [plan("loud")],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], ["at night"])],
                      "GeneratedAnswer": [generated(LOUD, unknowns=["caveat"])]})
        self.assertEqual((r.status, r.unknowns), ("partial", ("at night",)))

    def test_second_rewrite_adds_evidence_and_never_reassesses_identical_input(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery"),
                                             assess("partial", ["C-r00000-c00"], ["days"], rewrite="two days lasts"),
                                             assess("answered", ["C-r00000-c00"])],
                      "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]},
                     product="C", max_rewrites=2)
        prompts = [c[2] for c in self.calls("EvidenceAssessment")]
        self.assertEqual(len(prompts), len(set(prompts)))
        for entry in (t for t in r.trace if t.node == "rewrite" and "new_candidates" in t.detail):
            added = int(entry.detail.rsplit("=", 1)[1])
            self.assertGreater(added, 0)

    def test_same_rewrite_query_not_reused_with_two_rewrites_allowed(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery"),
                                             assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery")],
                      "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]},
                     product="C", max_rewrites=2)
        self.assertEqual(r.rewrites_used, 1)

    def test_degraded_retrieval_continues_as_degraded(self):
        deps = RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(fail=True),
                             bm25_cache=BM25Cache(), params=RetrievalParams(evidence_k=2, full_context_max_tokens=0))
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD)]}, deps=deps)
        self.assertEqual((r.status, r.service_status), ("answered", "degraded"))

    def _patched_retrieve(self, second):
        real, calls = workflow_nodes.retrieve, []

        def fake(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                return second(real, *args, **kwargs)
            return real(*args, **kwargs)
        return patch.object(workflow_nodes, "retrieve", side_effect=fake)

    def test_rewrite_retrieval_failure_keeps_candidates_and_degrades(self):
        def boom(real, *args, **kwargs):
            raise psycopg.OperationalError("connection lost")
        with self._patched_retrieve(boom):
            r = self.ask({"QueryPlan": [plan("noise")],
                          "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery")],
                          "GeneratedAnswer": [cite_first_candidate]}, product="C")
        self.assertEqual((r.service_status, r.status, r.rewrites_used), ("degraded", "partial", 1))
        self.assertEqual(len(self.calls("EvidenceAssessment")), 1)

    def test_rewrite_retrieval_degraded_is_propagated(self):
        def degraded(real, *args, **kwargs):
            return real(*args, **{**kwargs, "reranker": FakeReranker(fail=True)})
        with self._patched_retrieve(degraded):
            r = self.ask({"QueryPlan": [plan("noise")],
                          "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery"),
                                                 assess("answered", ["C-r00000-c00"])],
                          "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]}, product="C")
        self.assertEqual(r.service_status, "degraded")

    def test_read_error_in_retrieve_is_service_failure(self):
        with patch.object(workflow_nodes, "retrieve", side_effect=psycopg.OperationalError("db down")):
            r = self.ask({"QueryPlan": [plan("loud")]})
        self.assertEqual((r.status, r.service_status), (None, "failed"))
        self.assertIn("OperationalError", r.trace[-1].detail)

    def test_retrieval_snapshots_recorded(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD)]})
        self.assertEqual(len(r.retrievals), 1)
        snap = r.retrievals[0]
        self.assertEqual((snap.query, snap.strategy, snap.service_status), ("loud", "hybrid", "ok"))
        self.assertEqual(snap.candidates[0]["chunk_id"], "A-r00001-c00")
        self.assertIsInstance(snap.candidates[0]["dense_rank"], int)
        self.assertEqual(snap.candidates[0]["bm25_rank"], 1)
        self.assertNotIn("text", snap.candidates[0])
        self.assertEqual({c.name for c in snap.channels}, {"dense", "bm25", "rerank"})

    def test_rewrite_adds_second_snapshot(self):
        r = self.ask({"QueryPlan": [plan("noise")],
                      "EvidenceAssessment": [assess("partial", ["C-r00000-c00"], ["battery"], rewrite="battery"),
                                             assess("answered", ["C-r00000-c00"])],
                      "GeneratedAnswer": [generated(("Loud at night.", ["C-r00000-c00"]))]}, product="C")
        self.assertEqual([s.query for s in r.retrievals], ["noise", "battery"])

    def test_stream_yields_trace_then_result(self):
        llm = FakeLLM({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                       "GeneratedAnswer": [generated(LOUD)]})
        items = list(stream_answer(self.conn, "A", "Is it loud?", llm=llm, retrieval=self.deps))
        entries, result = items[:-1], items[-1]
        self.assertTrue(all(isinstance(e, TraceEntry) for e in entries))
        self.assertEqual([e.node for e in entries], ["plan", "retrieve", "assess", "generate", "verify"])
        self.assertIsInstance(result, QAResult)
        self.assertEqual((result.status, result.trace), ("answered", tuple(entries)))

    def test_stream_validates_on_first_next(self):
        gen = stream_answer(self.conn, "ZZZ", "Is it loud?", llm=FakeLLM({}), retrieval=self.deps)
        with self.assertRaises(ProductNotFoundError):
            next(gen)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            self.ask({}, question="   ")
        with self.assertRaises(ProductNotFoundError):
            self.ask({}, product="ZZZ")


if __name__ == "__main__":
    unittest.main()
