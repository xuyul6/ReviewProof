import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewproof.eval.gold import GoldLabel
from reviewproof.eval.rerank_check import auc, auc_interval, best_cut, main, score_question, summarize
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow.state import RetrievalDeps
from tests.support import (TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, FakeReranker, make_manifest,
                           make_product, write_jsonl)

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)]
BATTERY = [f"Battery review {i}: the battery lasts two days." for i in range(6)]
EXACT = "Battery review 3: the battery lasts two days."


class AucTests(unittest.TestCase):
    def test_perfect_reversed_and_tied(self):
        self.assertEqual(auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(auc([0.1], [0.9]), 0.0)
        self.assertEqual(auc([0.5, 0.5], [0.5]), 0.5)

    def test_counts_pairs(self):
        # 0.9 beats both negatives, 0.3 beats only 0.2: 3 of 4 pairs.
        self.assertEqual(auc([0.9, 0.3], [0.5, 0.2]), 0.75)

    def test_interval_is_deterministic_and_brackets_the_point(self):
        pos, neg = [0.9, 0.7, 0.4, 0.3], [0.5, 0.2, 0.1]
        first, second = auc_interval(pos, neg), auc_interval(pos, neg)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lo"], first["auc"])
        self.assertLessEqual(first["auc"], first["hi"])
        self.assertEqual((first["n_pos"], first["n_neg"]), (4, 3))

    def test_interval_needs_both_classes(self):
        self.assertIsNone(auc_interval([0.4], []))


class BestCutTests(unittest.TestCase):
    def test_lowest_cut_with_the_best_accuracy(self):
        self.assertEqual(best_cut([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), (0.8, 1.0))

    def test_refusing_everything_is_a_candidate(self):
        self.assertEqual(best_cut([0.5, 0.6], [False, False]), (math.inf, 1.0))


def row(qid, gold, top1, strategy="hybrid", status="ok"):
    return {"qid": qid, "gold": gold, "top1": top1, "strategy": strategy, "service_status": status}


def wf(answered, silver):
    return {"answered_binary": answered, "answerable_silver": silver}


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        rows = [row("a", "answered", 0.9), row("b", "partial", 0.6, "full_context"), row("c", "partial", 0.05),
                row("d", "unanswerable", 0.1), row("e", "unanswerable", 0.02, "full_context"),
                row("f", "answered", None, status="failed")]
        workflow = {"a": wf(True, True), "b": wf(True, False), "c": wf(False, True), "d": wf(False, False),
                    "e": wf(True, False), "f": wf(True, True)}
        self.m = summarize(rows, workflow)

    def test_failed_scoring_is_left_out_of_everything(self):
        self.assertEqual((self.m["questions"], self.m["scored"], self.m["failed"]), (6, 5, 1))
        self.assertEqual(self.m["gold"], {"answered": 1, "partial": 2, "unanswerable": 2})

    def test_auc_counts_partial_as_answerable(self):
        self.assertAlmostEqual(self.m["auc_top1"]["all"]["auc"], 5 / 6)
        self.assertEqual(self.m["auc_top1"]["full_context"]["auc"], 1.0)

    def test_accuracy_on_the_same_questions(self):
        acc = self.m["accuracy"]
        self.assertEqual(acc["workflow"], 0.6)
        self.assertEqual(acc["b2_silver"], 0.8)
        self.assertEqual(acc["always_answerable"], 0.6)
        self.assertEqual(acc["best_cut_in_sample"], {"cut": 0.05, "accuracy": 0.8})

    def test_refusals(self):
        refusals = self.m["refusals"]
        self.assertEqual((refusals["n"], refusals["false"], refusals["correct"]), (2, 1, 1))
        self.assertEqual(refusals["false_by_gold"], {"partial": 1})
        self.assertEqual(refusals["auc_top1"]["auc"], 0.0)
        self.assertEqual(self.m["false_answers"], 1)


def gold(qid, product, answerability, grades):
    return GoldLabel(qid=qid, split="dev", product_id=product, question_type="descriptive",
                     answerability=answerability, verdict=None, grades=grades, draft=None, blind=None)


def question(qid, product, text):
    return Question(qid=qid, product_id=product, question_text=text, question_type="descriptive",
                    is_answerable=True, answers=({"answerText": "SECRET-ANSWER"},), review_snippets=())


class SampleData(DatabaseTestCase):
    """A is a full-context product, C a hybrid one."""

    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "C"),
                     {"A": make_product("A"), "C": make_product("C", NOISE + BATTERY)},
                     {"A": [question("qa", "A", "Does the battery last?")],
                      "C": [question("qc", "C", EXACT), question("qn", "C", "Is it waterproof?")]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        small = self.conn.execute("SELECT token_estimate FROM products WHERE product_id = 'A'").fetchone()[0]
        self.conn.rollback()
        self.params = RetrievalParams(full_context_max_tokens=small)

    def deps(self, reranker=None):
        return RetrievalDeps(embedder=FakeEmbedder(), embedding_model="fake", reranker=reranker or FakeReranker(),
                             bm25_cache=BM25Cache(), params=self.params)


class ScoreQuestionTests(SampleData):
    def test_full_context_product_scores_every_chunk(self):
        result = score_question(self.conn, gold("qa", "A", "answered", {"A-r00000-c00": 2}),
                                "Does the battery last?", self.deps())
        self.assertEqual((result["strategy"], result["service_status"], result["n_scored"]), ("full_context", "ok", 3))
        self.assertEqual(result["top"][0][0], "A-r00000-c00")
        self.assertEqual(result["top1"], result["top"][0][1])
        self.assertEqual(result["grade2_best_rank"], 1)

    def test_hybrid_product_scores_the_rerank_pool(self):
        result = score_question(self.conn, gold("qc", "C", "answered", {"C-r00009-c00": 2, "C-r00000-c00": 0}),
                                EXACT, self.deps())
        self.assertEqual((result["strategy"], result["service_status"], result["n_scored"]), ("hybrid", "ok", 12))
        self.assertEqual(result["top"][0][0], "C-r00009-c00")
        self.assertEqual((result["grade2_total"], result["grade2_best_rank"]), (1, 1))

    def test_reranker_failure_is_recorded_not_raised(self):
        for qid, product, text in (("qa", "A", "Does the battery last?"), ("qc", "C", EXACT)):
            result = score_question(self.conn, gold(qid, product, "answered", {}), text,
                                    self.deps(FakeReranker(fail=True)))
            self.assertEqual((result["service_status"], result["top1"], result["n_scored"]), ("failed", None, 0))


class RerankCheckCliTests(SampleData):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        base = {"label_set": "g", "guideline_version": "gold-v1", "dataset_id": "test-dataset", "split": "dev",
                "question_type": "descriptive", "verdict": None, "rationale": "r", "annotator": "human", "pool": [],
                "draft": None, "blind": None, "changed_from_draft": True}
        self.labels = write_jsonl(self.root / "labels.jsonl", [
            dict(base, qid="qa", product_id="A", answerability="answered", grades={"A-r00000-c00": 2}),
            dict(base, qid="qc", product_id="C", answerability="partial", grades={"C-r00009-c00": 2}),
            dict(base, qid="qn", product_id="C", answerability="unanswerable", grades={"C-r00000-c00": 0})])
        self.workflow = self.root / "per_question.jsonl"

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def run_main(self, qids=("qa", "qc", "qn")):
        write_jsonl(self.workflow, [{"qid": q, "system": s, "answered_binary": q != "qn", "answerable_silver": True}
                                    for q in qids for s in ("workflow", "b0_no_evidence")])
        argv = ["--labels", str(self.labels), "--split", "dev", "--workflow", str(self.workflow), "--name", "rc",
                "--out-root", str(self.root)]
        env = {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL,
               "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS": str(self.params.full_context_max_tokens)}
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, embedder_factory=lambda s: FakeEmbedder(), reranker_factory=lambda s: FakeReranker())
        return code, err.getvalue()

    def test_writes_scores_metrics_and_report(self):
        code, err = self.run_main()
        self.assertEqual(code, 0, err)
        out = self.root / "rc"
        rows = {r["qid"]: r for r in map(json.loads, (out / "per_question.jsonl").read_text().splitlines())}
        self.assertEqual({q: r["strategy"] for q, r in rows.items()},
                         {"qa": "full_context", "qc": "hybrid", "qn": "hybrid"})
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual((metrics["scored"], metrics["accuracy"]["workflow"]), (3, 1.0))
        config = json.loads((out / "config.json").read_text())
        for key in ("git_commit", "dataset_id", "rerank_model", "embedding_model", "retrieval_params", "labels_path",
                    "labels_sha256", "workflow_path", "workflow_sha256", "split", "started_at"):
            self.assertIn(key, config)
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("AUC", report)
        self.assertNotIn("SECRET", report + (out / "per_question.jsonl").read_text())

    def test_missing_workflow_row_exits_2(self):
        code, err = self.run_main(qids=("qa", "qc"))
        self.assertEqual(code, 2)
        self.assertIn("qn", err)
        self.assertFalse((self.root / "rc" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
