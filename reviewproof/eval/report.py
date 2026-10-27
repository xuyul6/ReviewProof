"""Aggregate evaluation rows into metrics.json and a Chinese report.md. Numbers only: no claims
of improvement beyond the table values and their intervals."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

from reviewproof.eval.metrics import STATES, answerability_metrics, bootstrap_ci, paired_diff_ci, three_state
from reviewproof.eval.retrieval_eval import VARIANTS

RETRIEVAL_METRICS = ("recall@5", "recall@10", "recall@20", "recall@40", "hit@5", "mrr@10", "ndcg@5")
SYSTEM_LABELS = {"workflow": "完整工作流", "b0_no_evidence": "B0 不给证据", "b1_full_context": "B1 全量上下文",
                 "workflow_b1_qids": "完整工作流（与 B1 同题）", "b2_amazonqa_classifier": "B2 AmazonQA 自带分类器"}
B2 = "b2_amazonqa_classifier"
TOKEN_BANDS = (("≤20k", 0, 20_000), ("20k–60k", 20_000, 60_000), (">60k", 60_000, None))
B1_SCOPE_NOTE = ("B1 只在混合检索商品的题上运行（全量上下文商品上它与工作流相同），所以和 B1 比较时请看"
                 "\"完整工作流（与 B1 同题）\"一行，它只统计 B1 也跑了的题。")

CAVEATS = """- **可答性银标**是 AmazonQA 分类器基于每题 10 个 BM25 片段做的判断，不是人工金标。换成完整评论后，部分"不可答"的题其实可答，所以对照银标的"误答"会被高估。
- **检索参考证据**是把 AmazonQA 为每题挑出的 10 个 BM25 片段对齐到评论块得到的，不代表这些块真的相关。检索分数衡量的是与 AmazonQA 的 BM25 结果的一致程度，天然偏向 BM25，不是人工金标。
- 人工金标评测在 5b 阶段进行。"""

GOLD_CAVEATS = """- **金标**由子 agent 按 docs/annotation_guidelines_cn.md 预标，再由人工逐题审核（通过或修改）得到；每题都经过人确认。
- **证据等级 0（无关）**主要来自预标：审核时预标为无关的块默认折叠，人工不一定逐块复核，所以漏标的相关块会让召回偏低。
- **候选池**是四种检索方案各取前 10 的并集，加上标注时搜索找到的池外块；池外没被看到的相关块不计入。
- **盲标一致率**是人和子 agent 之间的，不是两位人工之间的；定位文档 5.6 节要求的双人标注，这里只有一位人工。
- 子 agent 与被测系统同属 Claude 模型家族，审核能纠正明显的错误，但同类偏差未必都能发现。"""


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
    names = RETRIEVAL_METRICS + (("ndcg_graded@5",) if any("ndcg_graded@5" in (r.get(v) or {})
                                                          for r in included for v in VARIANTS) else ())
    for v in VARIANTS:
        ok = [r for r in included if r.get(v) is not None]
        variants[v] = {"n": len(ok), **{m: _interval([r[v][m] for r in ok]) for m in names}}
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
    mean_relevant = statistics.fmean(r["n_relevant"] for r in included) if included else None
    return {"n_total": len(rows), "n_included": len(included), "excluded": excluded, "mean_relevant": mean_relevant,
            "metrics": list(names),
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


def _failure_cases(rows: Sequence[dict], label_key: str = "answerable_silver") -> dict:
    def pick(pred):
        return [{"qid": r["qid"], "status": r["status"], label_key: r[label_key],
                 "answer": (r["answer_text"] or "")[:200]} for r in rows if pred(r)][:5]
    return {"false_refusal": pick(lambda r: r[label_key] and r["answered_binary"] is False),
            "answered_silver_unanswerable": pick(lambda r: not r[label_key] and r["answered_binary"]),
            "errors": [{"qid": r["qid"], "error": r["error"]} for r in rows if r.get("error")][:5]}


def _verdict_accuracy(rows: Sequence[dict]) -> dict:
    """Yes/no questions whose gold verdict is yes, no or mixed; a missing system verdict counts as wrong."""
    scored = [r for r in rows if r.get("verdict_gold") in ("yes", "no", "mixed") and r["service_status"] != "failed"]
    return {"n": len(scored), "accuracy": _interval([1.0 if r["verdict"] == r["verdict_gold"] else 0.0 for r in scored])}


def _token_bands(by_system: dict[str, list[dict]]) -> list[dict]:
    """Workflow vs B1 three-state agreement with gold, on the same hybrid questions, by product size."""
    b1 = {r["qid"]: r for r in by_system.get("b1_full_context", []) if r["service_status"] != "failed"}
    wf = {r["qid"]: r for r in by_system.get("workflow", []) if r["service_status"] != "failed" and r["qid"] in b1}
    out = []
    for band, lo, hi in TOKEN_BANDS:
        qids = [q for q, r in wf.items() if r.get("product_tokens") is not None
                and lo < r["product_tokens"] <= (hi if hi is not None else float("inf"))]
        if not qids:
            continue
        hit = lambda r: 1.0 if r["status"] == r["answerability_gold"] else 0.0  # noqa: E731
        out.append({"band": band, "n": len(qids), "workflow": _interval([hit(wf[q]) for q in qids]),
                    "b1_full_context": _interval([hit(b1[q]) for q in qids])})
    return out


def _fmt(v, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _gold_sections(metrics: dict) -> list[str]:
    out = ["", "## 3b. 三态混淆矩阵与是非题结论", "",
           "行是金标，列是系统输出；服务故障不计入。三态一致率 = 系统三态与金标完全相同的比例。"]
    for system, m in metrics["three_state"].items():
        out += ["", f"**{SYSTEM_LABELS.get(system, system)}**（n={m['n']}，三态一致率 {_ci(m['agreement'])}）", "",
                "| 金标 \\ 系统 | " + " | ".join(STATES) + " |", "|---|" + "---|" * len(STATES)]
        out += [f"| {g} | " + " | ".join(str(m["confusion"][g][p]) for p in STATES) + " |" for g in STATES]
    out += ["", "是非题结论准确率（金标结论为 yes / no / mixed 的是非题；系统没给结论算错）：", "",
            "| 系统 | n | 准确率 |", "|---|---|---|"]
    out += [f"| {SYSTEM_LABELS.get(s, s)} | {m['n']} | {_ci(m['accuracy'])} |"
            for s, m in metrics["verdict_accuracy"].items()]
    ag = metrics.get("agreement")
    out += ["", "## 3c. 一致率", ""]
    if ag:
        out += ["| 对比 | n | Cohen's kappa | 一致比例 | 备注 |", "|---|---|---|---|---|",
                f"| 金标 vs AmazonQA 银标（二值） | {ag['gold_vs_silver']['n']} | {_fmt(ag['gold_vs_silver'].get('kappa'))} | "
                f"{_fmt(ag['gold_vs_silver'].get('agreement'))} | 银标基于 10 个 BM25 片段 |",
                f"| 子 agent 预标 vs 人工终审（三态） | {ag['draft_vs_final']['n']} | {_fmt(ag['draft_vs_final'].get('kappa'))} | "
                f"{_fmt(ag['draft_vs_final'].get('agreement'))} | 终审改动了 {_fmt(ag['draft_vs_final'].get('changed_share'))} 的题"
                "（任一项不同即算） |",
                f"| 人工盲标 vs 子 agent 预标（三态） | {ag['blind_vs_draft']['n']} | {_fmt(ag['blind_vs_draft'].get('kappa'))} | "
                f"{_fmt(ag['blind_vs_draft'].get('agreement'))} | 人和子 agent 之间，不是两位人工之间 |"]
    else:
        out.append("本次未计算。")
    out += ["", "## 3d. 按评论量分段（混合检索商品，工作流 vs B1）", "",
            "只统计 B1 也跑了的题，按商品评论总 token 数分档；数值是三态一致率。用来判断全量上下文阈值该不该调。", ""]
    if metrics["token_bands"]:
        out += ["| 评论量 | n | 完整工作流 | B1 全量上下文 |", "|---|---|---|---|"]
        out += [f"| {b['band']} | {b['n']} | {_ci(b['workflow'])} | {_ci(b['b1_full_context'])} |"
                for b in metrics["token_bands"]]
    else:
        out.append("没有可比的题。")
    return out


def _markdown(metrics: dict) -> str:
    cfg = metrics["config"]
    out = [f"# 评测报告：{cfg.get('name', '')}", "", "## 1. 概要", ""]
    out += [f"- {k}: `{v}`" for k, v in cfg.items()]
    gold = metrics.get("labels") == "gold"
    out += ["", GOLD_CAVEATS if gold else CAVEATS, "", "## 2. 检索", ""]
    ret = metrics["retrieval"]
    if ret is None:
        out.append("本次未运行。")
    else:
        out.append(f"共 {ret['n_total']} 题，计入指标 {ret['n_included']} 题；排除：{ret['excluded'] or '无'}；"
                   f"各方案出错次数：{ret['errors']}。每格为 均值 [95% 置信区间]。")
        names = ret["metrics"]
        out += ["", "| 方案 | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for v in VARIANTS:
            row = ret["variants"][v]
            out.append(f"| {v} | {row['n']} | " + " | ".join(_ci(row[m]) for m in names) + " |")
        if ret["mean_relevant"]:
            out += ["", f"每题平均有 {ret['mean_relevant']:.1f} 个参考证据块，所以 Recall@5 的上限约为 "
                        f"{min(1.0, 5 / ret['mean_relevant']):.2f}，不要把接近这个值的分数读成检索偏弱。"]
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
        out.append(f"对照{'人工金标' if gold else ' AmazonQA 银标'}。"
                   "\"答了\" = answered 或 partial；\"拒答\" = unanswerable 或不在服务范围。服务故障不进入分母，单独计数。")
        if gold:
            out += ["", "B2 AmazonQA 自带分类器：直接拿银标当作预测，和金标对比（定位文档第 6 节的 B2 基线）。"]
        if "b1_full_context" in ans:
            out += ["", B1_SCOPE_NOTE]
        out += ["", "| 系统 | 计入/总数 | 服务故障 | 正确拒答率 | 误拒率 | 答复 F1 | 准确率 | 三态分布 | 服务故障被报成无法确认 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for system, m in ans.items():
            out.append(f"| {SYSTEM_LABELS.get(system, system)} | {m['n_scored']}/{m['n']} | {m['n_service_failed']} | "
                       f"{_ci(m['correct_refusal_rate'])} | {_ci(m['false_refusal_rate'])} | {_ci(m['answer_f1'])} | "
                       f"{_ci(m['accuracy'])} | {m['status_counts']} | {m['service_failure_reported_as_unanswerable']} |")
    if gold and ans is not None:
        out += _gold_sections(metrics)
    out += ["", "## 4. 成本与延迟", ""]
    cl = metrics["cost_latency"]
    if cl is None:
        out.append("本次未运行。")
    else:
        if "b1_full_context" in cl:
            out += [B1_SCOPE_NOTE, ""]
        out += ["| 系统 | n | 总成本 (USD) | 每题成本 | P50 延迟 (s) | P95 延迟 (s) | 平均 LLM 调用 | 降级率 | 越界引用 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for system, m in cl.items():
            fmt = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
            out.append(f"| {SYSTEM_LABELS.get(system, system)} | {m['n']} | {fmt(m['cost_usd_total'], 4)} | "
                       f"{fmt(m['cost_usd_mean'], 5)} | {fmt(m['latency_ms_p50'] and m['latency_ms_p50'] / 1000, 1)} | "
                       f"{fmt(m['latency_ms_p95'] and m['latency_ms_p95'] / 1000, 1)} | {fmt(m['llm_calls_mean'], 1)} | "
                       f"{fmt(m['degraded_rate'])} | {m['citation_out_of_product']} |")
        out.append("")
        out.append("成本按 Claude Haiku 5.5 价格（2026-10-27）计算，每次调用按它自己的输入量选价格档（超过 10 万 token 为长上下文价格）。")
    out += ["", "## 5. 失败案例", ""]
    fc = metrics["failure_cases"]
    if fc is None:
        out.append("本次未运行。")
    else:
        if gold:
            out.append("以下为完整工作流的案例（每类最多 5 题），对照人工金标。")
            titles = (("误拒（金标可答，系统拒答）", "false_refusal"),
                      ("误答（金标不可答，系统答了）", "answered_silver_unanswerable"))
        else:
            out.append("以下为完整工作流的案例（每类最多 5 题）。对照银标的\"误答\"需人工核对：银标基于片段，完整评论可能确实能回答。")
            titles = (("误拒（银标可答，系统拒答）", "false_refusal"),
                      ("对照银标的误答（银标不可答，系统答了）——需人工核对", "answered_silver_unanswerable"))
        for title, key in titles:
            out += ["", f"**{title}**", ""]
            out += [f"- `{c['qid']}` · {c['status']} · {c['answer']!r}" for c in fc[key]] or ["- 无"]
        if fc["errors"]:
            out += ["", "**运行出错**", ""] + [f"- `{c['qid']}` · {c['error']}" for c in fc["errors"]]
    out += ["", "## 6. 局限与下一步", ""]
    if gold:
        out += ["- 金标每个 split 约 100 题，区间仍然较宽；结论只写表中的数值和区间。",
                "- 金标的局限见第 1 节；结论支持率（逐条核对引用）不在本报告里。",
                "- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。", ""]
    else:
        out += ["- 本报告是小规模试跑，样本量小，区间较宽；不据此下\"提升了多少\"的结论。",
                "- 可答性和检索参考证据都来自 AmazonQA 银标，不是人工金标；5b 阶段做约 200 题人工金标后再给出结论。",
                "- 全量上下文商品的题不进入检索指标，只出现在可答性评测里。", ""]
    return "\n".join(out)


def build_report(*, retrieval_rows: Sequence[dict] | None, workflow_rows: Sequence[dict] | None,
                 config: dict, out_dir: Path, labels: str = "silver", agreement: dict | None = None) -> dict:
    """labels="gold": workflow rows carry answerability_gold / answerable_gold / verdict_gold /
    product_tokens, and the report adds B2, three-state confusion, verdict accuracy, agreement and
    the token-band comparison."""
    gold = labels == "gold"
    label_key = "answerable_gold" if gold else "answerable_silver"
    by_system: dict[str, list[dict]] = {}
    for r in workflow_rows or []:
        by_system.setdefault(r["system"], []).append(r)
    if "b1_full_context" in by_system and "workflow" in by_system:
        b1_qids = {r["qid"] for r in by_system["b1_full_context"]}
        by_system["workflow_b1_qids"] = [r for r in by_system["workflow"] if r["qid"] in b1_qids]
    if gold and "workflow" in by_system:
        by_system[B2] = [dict(r, system=B2, status="answered" if r["answerable_silver"] else "unanswerable",
                              answered_binary=r["answerable_silver"], service_status="ok")
                         for r in by_system["workflow"]]
    real = {s: rows for s, rows in by_system.items() if s != B2}
    metrics = {
        "config": config,
        "retrieval": _retrieval_summary(retrieval_rows) if retrieval_rows else None,
        "labels": labels,
        "answerability": ({s: answerability_metrics(rows, label_key) for s, rows in by_system.items()}
                          if workflow_rows else None),
        "cost_latency": {s: _cost_latency(rows) for s, rows in real.items()} if workflow_rows else None,
        "failure_cases": _failure_cases(by_system.get("workflow", []), label_key) if workflow_rows else None,
    }
    if gold and workflow_rows:
        metrics.update(
            three_state={s: three_state(rows) for s, rows in real.items()},
            verdict_accuracy={s: _verdict_accuracy(rows) for s, rows in real.items() if s != "b0_no_evidence"},
            token_bands=_token_bands(by_system), agreement=agreement)
    metrics = json.loads(json.dumps(metrics, ensure_ascii=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(metrics), encoding="utf-8")
    return metrics
