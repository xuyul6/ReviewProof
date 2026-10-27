import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.llm import LLMError
from reviewproof.workflow.cli import main
from reviewproof.workflow.schemas import Claim, EvidenceAssessment, GeneratedAnswer, QueryPlan
from tests.support import (TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest,
                           make_product)

QUESTION = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno",
                    is_answerable=True, answers=({"answerText": "SECRET-ANSWER"},),
                    review_snippets=("SECRET-SNIPPET",))


def answered_script():
    return {"QueryPlan": [QueryPlan(route="review_question", out_of_scope_reason="", is_yes_no=True,
                                    search_query="loud")],
            "EvidenceAssessment": [EvidenceAssessment(status="answered", supporting_chunk_ids=["A-r00001-c00"],
                                                      missing=[], rewrite_query="")],
            "GeneratedAnswer": [GeneratedAnswer(claims=[Claim(text="It is too loud.", chunk_ids=["A-r00001-c00"])],
                                                unknowns=[], verdict="yes")]}


class WorkflowCliTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Cheap plastic."])}, {"A": [QUESTION]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)

    def run_main(self, argv, script=None):
        out, err = io.StringIO(), io.StringIO()
        self.llm = FakeLLM(script if script is not None else answered_script())
        with patch.dict(os.environ, {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, llm_factory=lambda s: self.llm, embedder_factory=lambda s: FakeEmbedder(),
                        reranker_factory=lambda s: FakeReranker())
        return code, out.getvalue(), err.getvalue()

    def test_answered_prints_answer_citations_and_trace(self):
        code, out, _ = self.run_main(["--product", "A", "--question", "Is it loud?"])
        self.assertEqual(code, 0)
        self.assertIn("It is too loud. [1]", out)
        self.assertIn("status=answered", out)
        self.assertIn("A-r00001-c00", out)
        self.assertIn("plan", out)

    def test_qid_json_never_prints_community_data(self):
        code, out, _ = self.run_main(["--qid", "q1", "--json"])
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual((result["product_id"], result["question"], result["status"]), ("A", "Is it loud?", "answered"))
        self.assertNotIn("SECRET", out)

    def test_service_failure_exits_1(self):
        script = answered_script()
        script["EvidenceAssessment"] = [LLMError("down")]
        code, out, _ = self.run_main(["--product", "A", "--question", "Is it loud?"], script)
        self.assertEqual(code, 1)
        self.assertIn("service error", out)

    def test_unknown_product_qid_mismatch_and_blank_question_exit_2(self):
        self.assertEqual(self.run_main(["--product", "ZZZ", "--question", "Is it loud?"])[0], 2)
        self.assertEqual(self.run_main(["--qid", "q1", "--product", "B"])[0], 2)
        self.assertEqual(self.run_main(["--qid", "nope"])[0], 2)
        code, _, err = self.run_main(["--product", "A", "--question", "   "])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
