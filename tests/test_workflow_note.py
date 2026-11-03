import unittest
from unittest.mock import patch

from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.llm import LLMError
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow import RetrievalDeps, answer_question
from reviewproof.workflow import nodes as workflow_nodes
from reviewproof.workflow.answer import Note
from reviewproof.workflow.schemas import GeneralNote
from tests.support import (REVIEW_TEXTS, DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest,
                           make_product)
from tests.test_workflow import LOUD, assess, generated, plan

SUGGESTION_EN = "Not from the reviews: where you could check this."


def note(kind="suggestion", text="Check the product page."):
    return GeneralNote(kind=kind, text=text)


class GeneralNoteTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self.deps = RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=FakeReranker(),
                                  bm25_cache=BM25Cache(),
                                  params=RetrievalParams(evidence_k=2, full_context_max_tokens=0))

    def ask(self, script, question="Is it loud?", product="A", advise=True):
        self.llm = FakeLLM(script)
        return answer_question(self.conn, product, question, llm=self.llm, retrieval=self.deps, advise=advise)

    def calls(self, name):
        return [c for c in self.llm.calls if c[0] == name]

    def unanswerable(self, **extra):
        return {"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("unanswerable")], **extra}

    def test_unanswerable_run_gets_a_note(self):
        r = self.ask(self.unanswerable(GeneralNote=[note()]))
        self.assertEqual(r.status, "unanswerable")
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")
        self.assertEqual(r.note, Note("suggestion", SUGGESTION_EN, "Check the product page."))
        self.assertEqual((r.trace[-1].node, r.trace[-1].status), ("advise", "ok"))
        self.assertEqual(r.llm_calls, 3)

    def test_partial_note_prompt_lists_only_the_missing_parts(self):
        r = self.ask({"QueryPlan": [plan("loud", yes_no=False)],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"], missing=["battery life"])],
                      "GeneratedAnswer": [generated(LOUD, verdict=None)], "GeneralNote": [note("general", "Usually.")]},
                     question="How loud is it and how long does the battery last?")
        self.assertEqual(r.status, "partial")
        self.assertEqual(r.note.kind, "general")
        prompt = self.calls("GeneralNote")[0][2]
        self.assertIn("The reviews did not cover: battery life", prompt)

    def test_partial_without_missing_parts_gets_no_note(self):
        r = self.ask({"QueryPlan": [plan("loud", yes_no=False)],
                      "EvidenceAssessment": [assess("partial", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD, verdict=None)]})
        self.assertEqual(r.status, "partial")
        self.assertIsNone(r.note)
        self.assertEqual(self.calls("GeneralNote"), [])

    def test_answered_run_gets_no_note(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [assess("answered", ["A-r00001-c00"])],
                      "GeneratedAnswer": [generated(LOUD)]})
        self.assertEqual(r.status, "answered")
        self.assertIsNone(r.note)
        self.assertEqual(self.calls("GeneralNote"), [])

    def test_out_of_scope_run_gets_a_note_with_the_reason(self):
        r = self.ask({"QueryPlan": [plan("price", route="out_of_scope", reason="price is not in reviews")],
                      "GeneralNote": [note()]}, question="How much is it?")
        self.assertEqual((r.route, r.status), ("out_of_scope", "unanswerable"))
        self.assertEqual(r.note.text, "Check the product page.")
        self.assertIn("Outside what reviews can answer: price is not in reviews", self.calls("GeneralNote")[0][2])

    def test_service_failure_gets_no_note(self):
        r = self.ask({"QueryPlan": [plan("loud")], "EvidenceAssessment": [LLMError("down")]})
        self.assertEqual(r.service_status, "failed")
        self.assertIsNone(r.note)
        self.assertEqual(self.calls("GeneralNote"), [])

    def test_note_failure_keeps_the_answer(self):
        r = self.ask(self.unanswerable(GeneralNote=[LLMError("down")]))
        self.assertIsNone(r.note)
        self.assertEqual((r.status, r.service_status), ("unanswerable", "ok"))
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")
        self.assertEqual((r.trace[-1].node, r.trace[-1].status), ("advise", "failed"))

    def test_unexpected_note_error_keeps_the_answer(self):
        with patch.object(workflow_nodes, "advise_prompt", side_effect=TypeError("product row vanished")):
            r = self.ask(self.unanswerable())
        self.assertIsNone(r.note)
        self.assertEqual((r.status, r.service_status), ("unanswerable", "ok"))
        self.assertEqual(r.answer_text, "Verdict: Can't confirm\nThe reviews don't confirm this.")
        self.assertEqual((r.trace[-1].node, r.trace[-1].status), ("advise", "failed"))
        self.assertIn("TypeError", r.trace[-1].detail)

    def test_blank_note_text_is_dropped(self):
        r = self.ask(self.unanswerable(GeneralNote=[note(text="  ")]))
        self.assertIsNone(r.note)
        self.assertEqual((r.trace[-1].node, r.trace[-1].status), ("advise", "ok"))

    def test_note_is_off_by_default(self):
        r = self.ask(self.unanswerable(), advise=False)
        self.assertIsNone(r.note)
        self.assertEqual(self.calls("GeneralNote"), [])
        self.assertNotIn("advise", [t.node for t in r.trace])

    def test_prompt_has_title_category_and_no_reviews(self):
        self.conn.execute("UPDATE products SET title = 'Acme Desk Fan' WHERE product_id = 'A'")
        self.conn.commit()
        self.ask(self.unanswerable(GeneralNote=[note()]))
        prompt = self.calls("GeneralNote")[0][2]
        self.assertIn("Product: Acme Desk Fan (category: Electronics)", prompt)
        for text in REVIEW_TEXTS:
            self.assertNotIn(text, prompt)
        self.ask(self.unanswerable(GeneralNote=[note()]), product="B")
        prompt = self.calls("GeneralNote")[0][2]
        self.assertIn("Product: title unknown (category: Electronics)", prompt)
        self.assertNotIn("Cheap plastic", prompt)

    def test_answer_is_the_same_with_and_without_note(self):
        without = self.ask(self.unanswerable(), advise=False)
        with_note = self.ask(self.unanswerable(GeneralNote=[note()]))
        for field in ("status", "verdict", "answer_text", "claims", "citations", "unknowns", "service_status"):
            self.assertEqual(getattr(without, field), getattr(with_note, field), field)


if __name__ == "__main__":
    unittest.main()
