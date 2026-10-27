"""Aggregate evaluation rows into metrics.json and a Chinese report.md. Numbers only: no claims
of improvement beyond the table values and their intervals."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

from reviewproof.eval.metrics import answerability_metrics, bootstrap_ci, paired_diff_ci
from reviewproof.eval.retrieval_eval import VARIANTS

RETRIEVAL_METRICS = ("recall@5", "recall@10", "recall@20", "recall@40", "hit@5", "mrr@10", "ndcg@5")
SYSTEM_LABELS = {"workflow": "完整工作流", "b0_no_evidence": "B0 不给证据", "b1_full_context": "B1 全量上下文"}

CAVEATS = """- **可答性银标**是 AmazonQA 分类器基于每题 10 个 BM25 片段做的判断，不是人工金标。换成完整评论后，部分"不可答"的题其实可答，所以对照银标的"误答"会被高估。
- **检索参考证据**是把 AmazonQA 为每题挑出的 10 个 BM25 片段对齐到评论块得到的，不代表这些块真的相关。检索分数衡量的是与 AmazonQA 的 BM25 结果的一致程度，天然偏向 BM25，不是人工金标。
- 人工金标评测在 5b 阶段进行。"""


def _ci(d: dict | None, digits: int = 3) -> str:
    if not d:
        return "—"
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _interval(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def _retrieval_summary(rows: Sequence[dict]) -> dict:
    included = [r for r in rows if r["included"]]
    excluded: dict[str, int] = {}
    for r in rows:
        if not r["included"]:
            excluded[r["exclude_reason"]] = excluded.get(r["exclude_reason"], 0) + 1
    variants = {}
    for v in VARIANTS:
        ok = [r for r in included if r.get(v) is not None]
        variants[v] = {"n": len(ok), **{m: _interval([r[v][m] for r in ok]) for m in RETRIEVAL_METRICS}}
    diffs = {}
    for other in ("bm25", "dense", "hybrid"):
        both = [r for r in included if r.get("hybrid_rerank") is not None and r.get(other) is not None]
        diffs[other] = {}
        for m in ("hit@5", "ndcg@5"):
            if both:
                mean, lo, hi = paired_diff_ci([r["hybrid_rerank"][m] for r in both], [r[other][m] for r in both])
                diffs[other][m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(both)}
            else:
                diffs[other][m] = None
    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in included}):
        by_category[cat] = {v: _interval([r[v]["hit@5"] for r in included if r["category"] == cat and r.get(v)])
                            for v in VARIANTS}
    return {"n_total": len(rows), "n_included": len(included), "excluded": excluded,
            "errors": {v: sum(1 for r in included if r.get(v) is None) for v in VARIANTS},
            "variants": variants, "hybrid_rerank_minus": diffs, "hit@5_by_category": by_category}


def _cost_latency(rows: Sequence[dict]) -> dict:
    latencies = [r["latency_ms"] for r in rows]
    scored = [r for r in rows if r["service_status"] != "failed"]
    return {"n": len(rows), "cost_usd_total": sum(r["cost_usd"] for r in rows),
            "cost_usd_mean": statistics.fmean(r["cost_usd"] for r in rows) if rows else None,
            "latency_ms_p50": _pct(latencies, 0.5), "latency_ms_p95": _pct(latencies, 0.95),
            "llm_calls_mean": statistics.fmean(r["llm_calls"] for r in rows) if rows else None,
            "degraded_rate": (sum(r["service_status"] == "degraded" for r in scored) / len(scored)) if scored else None,
            "citation_out_of_product": sum(r.get("citation_out_of_product", 0) for r in rows)}


def _failure_cases(rows: Sequence[dict]) -> dict:
    def pick(pred):
        return [{"qid": r["qid"], "status": r["status"], "answerable_silver": r["answerable_silver"],
                 "answer": (r["answer_text"] or "")[:200]} for r in rows if pred(r)][:5]
    return {"false_refusal": pick(lambda r: r["answerable_silver"] and r["answered_binary"] is False),
            "answered_silver_unanswerable": pick(lambda r: not r["answerable_silver"] and r["answered_binary"]),
            "errors": [{"qid": r["qid"], "error": r["error"]} for r in rows if r.get("error")][:5]}


def _markdown(metrics: dict) -> str:
    cfg = metrics["config"]
    out = [f"# 评测报告：{cfg.get('name', '')}", "", "## 1. 概要", ""]
    out += [f"- {k}: `{v}`" for k, v in cfg.items()]
    out += ["", CAVEATS, "", "## 2. 检索", ""]
    ret = metrics["retrieval"]
    if ret is None:
        out.append("本次未运行。")
    else:
        out.append(f"共 {ret['n_total']} 题，计入指标 {ret['n_included']} 题；排除：{ret['excluded'] or '无'}；"
                   f"各方案出错次数：{ret['errors']}。每格为 均值 [95% 置信区间]。")
        out += ["", "| 方案 | n | " + " | ".join(RETRIEVAL_METRICS) + " |",
                "|---|---|" + "---|" * len(RETRIEVAL_METRICS)]
        for v in VARIANTS:
            row = ret["variants"][v]
            out.append(f"| {v} | {row['n']} | " + " | ".join(_ci(row[m]) for m in RETRIEVAL_METRICS) + " |")
        out += ["", "hybrid_rerank 减去其他方案（配对差值）：", "", "| 对比 | hit@5 | ndcg@5 |", "|---|---|---|"]
        for other, d in ret["hybrid_rerank_minus"].items():
            out.append(f"| hybrid_rerank − {other} | {_ci(d['hit@5'])} | {_ci(d['ndcg@5'])} |")
        out += ["", "按品类的 hit@5：", "", "| 品类 | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
        for cat, d in ret["hit@5_by_category"].items():
            out.append(f"| {cat} | " + " | ".join(_ci(d[v]) for v in VARIANTS) + " |")
    out += ["", "## 3. 可答性", ""]
    ans = metrics["answerability"]
    if ans is None:
        out.append("本次未运行。")
    else:
        out.append("\"答了\" = answered 或 partial；\"拒答\" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。")
        out += ["", "| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for system, m in ans.items():
            out.append(f"| {SYSTEM_LABELS.get(system, system)} | {m['n_scored']}/{m['n']} | {m['n_service_failed']} | "
                       f"{_ci(m['correct_refusal_rate'])} | {_ci(m['false_refusal_rate'])} | {_ci(m['answer_f1'])} | "
                       f"{_ci(m['accuracy'])} | {m['status_counts']} | {m['service_failure_reported_as_unanswerable']} |")
    out += ["", "## 4. 成本与延迟", ""]
    cl = metrics["cost_latency"]
    if cl is None:
        out.append("本次未运行。")
    else:
        out += ["| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for system, m in cl.items():
            fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
            out.append(f"| {SYSTEM_LABELS.get(system, system)} | {m['n']} | {fmt(m['cost_usd_total'], 4)} | "
                       f"{fmt(m['cost_usd_mean'], 5)} | {fmt(m['latency_ms_p50'] and m['latency_ms_p50'] / 1000, 1)} | "
                       f"{fmt(m['latency_ms_p95'] and m['latency_ms_p95'] / 1000, 1)} | {fmt(m['llm_calls_mean'], 1)} | "
                       f"{fmt(m['degraded_rate'])} | {m['citation_out_of_product']} |")
        out.append("")
        out.append("成本按 Claude Haiku 5.5 价格（2026-10-27）估算，每题按平均每次调用的输入量选价格档，是近似值。")
    out += ["", "## 5. 失败案例", ""]
    fc = metrics["failure_cases"]
    if fc is None:
        out.append("本次未运行。")
    else:
        out.append("以下为完整工作流的案例（每类最多 5 题）。对照银标的\"误答\"需人工核对：银标基于片段，完整评论可能确实能回答。")
        for title, key in (("误拒（银标可答，系统拒答）", "false_refusal"),
                           ("对照银标的误答（银标不可答，系统答了）——需人工核对", "answered_silver_unanswerable")):
            out += ["", f"**{title}**", ""]
            out += [f"- `{c['qid']}` · {c['status']} · {c['answer']!r}" for c in fc[key]] or ["- 无"]
        if fc["errors"]:
            out += ["", "**运行出错**", ""] + [f"- `{c['qid']}` · {c['error']}" for c in fc["errors"]]
    out += ["", "## 6. 局限与下一步", "",
            "- 本报告是小规模试跑，样本量小，区间较宽；不据此下\"提升了多少\"的结论。",
            "- 可答性和检索参考证据都来自 AmazonQA 银标，不是人工金标；5b 阶段做约 200 题人工金标后再给出结论。",
            "- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。", ""]
    return "\n".join(out)


def build_report(*, retrieval_rows: Sequence[dict] | None, workflow_rows: Sequence[dict] | None,
                 config: dict, out_dir: Path) -> dict:
    by_system: dict[str, list[dict]] = {}
    for r in workflow_rows or []:
        by_system.setdefault(r["system"], []).append(r)
    metrics = {
        "config": config,
        "retrieval": _retrieval_summary(retrieval_rows) if retrieval_rows else None,
        "answerability": {s: answerability_metrics(rows) for s, rows in by_system.items()} if workflow_rows else None,
        "cost_latency": {s: _cost_latency(rows) for s, rows in by_system.items()} if workflow_rows else None,
        "failure_cases": _failure_cases(by_system.get("workflow", [])) if workflow_rows else None,
    }
    metrics = json.loads(json.dumps(metrics, ensure_ascii=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(metrics), encoding="utf-8")
    return metrics
