import json
import tempfile
import unittest
from pathlib import Path

from reviewproof.eval.metrics import answerability_metrics, bootstrap_ci, paired_diff_ci
from reviewproof.eval.report import build_report


_DERIVE = object()


def wf(qid, silver, answered, system="workflow", status=_DERIVE, service="ok", latency=1000.0, cost=0.001):
    if status is _DERIVE:
        status = "answered" if answered else "unanswerable"
    return {"qid": qid, "system": system, "category": "Electronics", "question_type": "yesno",
            "strategy": "hybrid", "answerable_silver": silver, "answered_binary": answered, "status": status,
            "service_status": service, "route": "review_question", "verdict": None, "n_citations": 1,
            "citation_out_of_product": 0, "llm_calls": 3, "input_tokens": 100, "output_tokens": 10,
            "cost_usd": cost, "latency_ms": latency, "rewrites_used": 0, "retrieval_modes": ["dense+bm25+rerank"],
            "answer_text": f"answer for {qid}", "error": None}


ROWS = ([wf(f"t{i}", True, True) for i in range(3)] + [wf("fr", True, False), wf("cr", False, False),
        wf("oa", False, True), wf("sf", True, None, status=None, service="failed")])


def scores(v):
    return {"recall@5": v, "recall@10": v, "recall@20": v, "recall@40": v, "hit@5": v, "mrr@10": v, "ndcg@5": v}


RETRIEVAL_ROWS = [
    {"qid": "r1", "category": "Electronics", "question_type": "yesno", "strategy": "hybrid", "included": True,
     "exclude_reason": None, "n_relevant": 10, "latency_ms": 5.0, "bm25": scores(1.0), "dense": scores(0.0),
     "hybrid": scores(1.0), "hybrid_rerank": scores(1.0)},
    {"qid": "r2", "category": "Beauty", "question_type": "descriptive", "strategy": "hybrid", "included": True,
     "exclude_reason": None, "n_relevant": 10, "latency_ms": 5.0, "bm25": scores(0.0), "dense": None,
     "hybrid": None, "hybrid_rerank": None},
    {"qid": "r3", "category": "Beauty", "question_type": "descriptive", "strategy": "full_context",
     "included": False, "exclude_reason": "full_context", "n_relevant": 10},
]


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_ci_deterministic_and_bounds(self):
        values = [0, 1, 1, 0, 1, 1, 1, 0]
        self.assertEqual(bootstrap_ci(values), bootstrap_ci(values))
        mean, lo, hi = bootstrap_ci(values)
        self.assertAlmostEqual(mean, 5 / 8)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)
        self.assertEqual(bootstrap_ci([1, 1, 1]), (1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            bootstrap_ci([])

    def test_paired_diff_uses_same_resamples(self):
        self.assertEqual(paired_diff_ci([0, 1, 1, 0], [0, 1, 1, 0]), (0.0, 0.0, 0.0))
        self.assertEqual(paired_diff_ci([1, 1], [0, 0]), (1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            paired_diff_ci([1], [1, 0])


class AnswerabilityTests(unittest.TestCase):
    def test_metrics_exclude_service_failures(self):
        m = answerability_metrics(ROWS)
        self.assertEqual((m["n"], m["n_scored"], m["n_service_failed"]), (7, 6, 1))
        self.assertAlmostEqual(m["correct_refusal_rate"]["mean"], 0.5)
        self.assertAlmostEqual(m["false_refusal_rate"]["mean"], 0.25)
        self.assertAlmostEqual(m["answer_precision"], 0.75)
        self.assertAlmostEqual(m["answer_recall"], 0.75)
        self.assertAlmostEqual(m["answer_f1"]["mean"], 0.75)
        self.assertAlmostEqual(m["accuracy"]["mean"], 4 / 6)
        self.assertEqual(m["status_counts"], {"answered": 4, "unanswerable": 2, "service_failed": 1})
        self.assertEqual(m["service_failure_reported_as_unanswerable"], 0)

    def test_counts_service_failure_reported_as_unanswerable(self):
        bad = wf("x", True, None, status="unanswerable", service="failed")
        self.assertEqual(answerability_metrics([bad])["service_failure_reported_as_unanswerable"], 1)

    def test_empty_subset_rate_is_none(self):
        m = answerability_metrics([wf("a", True, True)])
        self.assertIsNone(m["correct_refusal_rate"])


class ReportTests(unittest.TestCase):
    def build(self, workflow_rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        metrics = build_report(retrieval_rows=RETRIEVAL_ROWS, workflow_rows=workflow_rows,
                               config={"name": "unit", "split": "dev", "seed": 20261027}, out_dir=out)
        return metrics, (out / "report.md").read_text(encoding="utf-8"), json.loads((out / "metrics.json").read_text())

    def test_report_files_and_sections(self):
        metrics, md, saved = self.build(ROWS + [wf("b0", True, True, system="b0_no_evidence")])
        for title in ("## 1. 概要", "## 2. 检索", "## 3. 可答性", "## 4. 成本与延迟", "## 5. 失败案例",
                      "## 6. 局限与下一步"):
            self.assertIn(title, md)
        for phrase in ("银标", "需人工核对", "BM25", "不是人工金标"):
            self.assertIn(phrase, md)
        self.assertEqual(saved["config"]["name"], "unit")
        self.assertEqual(saved["retrieval"]["variants"]["dense"]["n"], 1)
        self.assertEqual(saved["retrieval"]["errors"], {"bm25": 0, "dense": 1, "hybrid": 1, "hybrid_rerank": 1})
        self.assertEqual(saved["retrieval"]["excluded"], {"full_context": 1})
        self.assertIn("workflow", saved["answerability"])
        self.assertIn("b0_no_evidence", saved["answerability"])
        self.assertIn("fr", md)  # a false refusal is listed as a failure case
        self.assertEqual(metrics, saved)

    def test_workflow_compared_with_b1_on_the_same_questions(self):
        fc = dict(wf("fc", True, True), strategy="full_context")
        b1 = [wf("t0", True, False, system="b1_full_context"), wf("fr", True, False, system="b1_full_context")]
        _, md, saved = self.build(ROWS + [fc] + b1)
        matched = saved["answerability"]["workflow_b1_qids"]
        self.assertEqual(matched["n"], 2)  # only t0 and fr, not the full-context question
        self.assertEqual(saved["cost_latency"]["workflow_b1_qids"]["n"], 2)
        self.assertEqual(saved["answerability"]["workflow"]["n"], len(ROWS) + 1)
        self.assertIn("完整工作流（与 B1 同题）", md)
        self.assertIn("B1 只在混合检索商品的题上运行", md)
        lines = md.splitlines()
        for header in ("| 系统 | 计入/总数", "| 系统 | n | 总成本"):
            i = next(n for n, line in enumerate(lines) if line.startswith(header))
            self.assertTrue(all(line.startswith("|") for line in lines[i:i + 3]), header)  # table not split by the note

    def test_recall_ceiling_noted_next_to_retrieval_table(self):
        _, md, saved = self.build(None)
        self.assertAlmostEqual(saved["retrieval"]["mean_relevant"], 10.0)
        self.assertIn("Recall@5 的上限约为 0.50", md)

    def test_gold_report_has_b2_confusion_agreement_and_gold_caveats(self):
        def gold(r, ans, verdict=None, tokens=30_000):
            return dict(r, answerability_gold=ans, answerable_gold=ans != "unanswerable", verdict_gold=verdict,
                        product_tokens=tokens)
        rows = [gold(wf("g1", True, True), "answered", "yes"), gold(wf("g2", False, False), "unanswerable"),
                gold(wf("g3", True, False), "partial", tokens=90_000),
                gold(wf("g1", True, True, system="b1_full_context"), "answered", "yes"),
                gold(wf("g3", True, True, system="b1_full_context"), "partial", tokens=90_000)]
        rows[0]["verdict"] = "yes"
        agreement = {"gold_vs_silver": {"kappa": 0.5, "agreement": 0.8, "n": 3},
                     "draft_vs_final": {"kappa": 0.9, "changed_share": 0.1, "n": 3},
                     "blind_vs_draft": {"kappa": None, "agreement": 1.0, "n": 1}}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        metrics = build_report(retrieval_rows=None, workflow_rows=rows, config={"name": "gold"},
                               out_dir=Path(tmp.name), labels="gold", agreement=agreement)
        md = (Path(tmp.name) / "report.md").read_text(encoding="utf-8")
        b2 = metrics["answerability"]["b2_amazonqa_classifier"]
        self.assertEqual((b2["n"], b2["accuracy"]["mean"]), (3, 1.0))  # silver happens to match gold here
        self.assertEqual(metrics["three_state"]["workflow"]["confusion"]["partial"]["unanswerable"], 1)
        self.assertEqual(metrics["verdict_accuracy"]["workflow"]["n"], 1)
        self.assertEqual([b["band"] for b in metrics["token_bands"]], ["20k–60k", ">60k"])
        for phrase in ("人工逐题审核", "B2 AmazonQA 自带分类器", "三态混淆矩阵", "一致率", "是非题结论", "按评论量分段",
                       "不是两位人工"):
            self.assertIn(phrase, md)
        self.assertNotIn("AmazonQA 分类器基于每题 10 个 BM25 片段做的判断，不是人工金标", md)
        self.assertNotIn("b2_amazonqa_classifier", metrics["cost_latency"])

    def test_report_without_workflow_rows(self):
        _, md, saved = self.build(None)
        self.assertIn("本次未运行", md)
        self.assertIsNone(saved["answerability"])


if __name__ == "__main__":
    unittest.main()
