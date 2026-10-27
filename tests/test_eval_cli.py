import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewproof.eval.baselines import B0Answer
from reviewproof.eval.cli import main
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.workflow.schemas import Claim, EvidenceAssessment, GeneratedAnswer, QueryPlan
from tests.support import (TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, FakeLLM, FakeReranker, make_manifest,
                           make_product)

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)]


def first_chunk(prompt):
    return re.search(r"\[([A-Z]+-r\d{5}-c\d{2})\]", prompt).group(1)


def script(n=20):
    return {"QueryPlan": [lambda p: QueryPlan(route="review_question", out_of_scope_reason="", is_yes_no=False,
                                              search_query="loud")] * n,
            "EvidenceAssessment": [lambda p: EvidenceAssessment(status="answered",
                                                                supporting_chunk_ids=[first_chunk(p)], missing=[],
                                                                rewrite_query="")] * n,
            "GeneratedAnswer": [lambda p: GeneratedAnswer(claims=[Claim(text="Loud.", chunk_ids=[first_chunk(p)])],
                                                          unknowns=[], verdict=None)] * n,
            "B0Answer": [lambda p: B0Answer(answered=True, answer="Probably.")] * n}


def question(qid, product, snippet, answerable=True):
    return Question(qid=qid, product_id=product, question_text="Is the fan loud?", question_type="descriptive",
                    is_answerable=answerable, answers=({"answerText": "SECRET-ANSWER"},), review_snippets=(snippet,))


class EvalCliTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "C"), {"A": make_product("A"), "C": make_product("C", NOISE)},
                     {"A": [question("qa", "A", "Too loud for me.")],
                      "C": [question("qc", "C", "Noise review 2: the fan is loud"),
                            question("qn", "C", "the fan is loud at night", answerable=False)]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def run_main(self, argv, llm):
        self.llm = llm
        env = {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL, "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS": "20"}
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv + ["--out-root", str(self.root)], llm_factory=lambda s: llm,
                        embedder_factory=lambda s: FakeEmbedder(), reranker_factory=lambda s: FakeReranker())
        return code, out.getvalue(), err.getvalue()

    def test_end_to_end_small(self):
        code, _, err = self.run_main(["--split", "dev", "--name", "t1", "--sample", "10"], FakeLLM(script()))
        self.assertEqual(code, 0, err)
        out = self.root / "t1"
        for name in ("config.json", "retrieval.jsonl", "per_question.jsonl", "metrics.json", "report.md"):
            self.assertTrue((out / name).exists(), name)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("## 2. 检索", report)
        self.assertIn("## 3. 可答性", report)
        config = json.loads((out / "config.json").read_text())
        for key in ("git_commit", "dataset_id", "source_sha256", "embedding_model", "rerank_model", "llm_model",
                    "retrieval_params", "max_rewrites", "seed", "sample", "split", "started_at"):
            self.assertIn(key, config)
        self.assertNotIn("approximate", config["cost_note"])  # each call is priced at its own tier
        retrieval = [json.loads(l) for l in (out / "retrieval.jsonl").read_text().splitlines()]
        self.assertEqual({r["qid"]: r["exclude_reason"] for r in retrieval},
                         {"qa": "full_context", "qc": None, "qn": "silver_unanswerable"})
        for _, system, prompt in self.llm.calls:
            self.assertNotIn("SECRET", system + prompt)

    def test_retrieval_only_makes_no_llm_calls(self):
        code, _, _ = self.run_main(["--split", "dev", "--name", "t2", "--retrieval-only"], FakeLLM({}))
        self.assertEqual(code, 0)
        self.assertEqual(self.llm.calls, [])
        self.assertFalse((self.root / "t2" / "per_question.jsonl").exists())
        self.assertIn("本次未运行", (self.root / "t2" / "report.md").read_text(encoding="utf-8"))

    def test_rerun_resumes(self):
        self.run_main(["--split", "dev", "--name", "t3", "--systems", "workflow"], FakeLLM(script()))
        before = (self.root / "t3" / "per_question.jsonl").read_text()
        code, _, _ = self.run_main(["--split", "dev", "--name", "t3", "--systems", "workflow"], FakeLLM({}))
        self.assertEqual(code, 0)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual((self.root / "t3" / "per_question.jsonl").read_text(), before)

    def test_unknown_system_exits_2(self):
        code, _, err = self.run_main(["--split", "dev", "--name", "t4", "--systems", "nope"], FakeLLM({}))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
