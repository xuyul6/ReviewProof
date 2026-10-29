"""python -m reviewproof.eval.rerank_check --labels annotations/gold_v1/labels.jsonl --split dev \
       --workflow reports/eval/gold-dev/per_question.jsonl --name rerank-check-dev

Can the reranker's top score tell answerable questions from unanswerable ones? Checked before deciding
whether phase 6 should train an answerability model; no LLM calls. Each gold question of the split is
scored against what the reranker sees in the system: every chunk of a full-context product, or the
hybrid pool (rerank_pool chunks) of a larger one. The query is the question text, not the workflow's
planned query. The top score is compared with gold answerability (answered and partial count as
answerable), and with the workflow and B2 (the silver label) from an earlier gold eval of the same questions.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import psycopg

from reviewproof.config import Settings, load_settings
from reviewproof.db import connect
from reviewproof.embedding import Embedder
from reviewproof.eval.cli import REPO_ROOT, _git_commit, _write_jsonl
from reviewproof.eval.gold import GoldError, GoldLabel, load_gold
from reviewproof.eval.metrics import RESAMPLES, SEED, _percentile
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.cli import default_embedder, default_reranker
from reviewproof.retrieval.rerank import Reranker
from reviewproof.retrieval.retriever import retrieve
from reviewproof.workflow.state import RetrievalDeps

TOP_KEPT = 10
STRATEGIES = ("hybrid", "full_context")


def score_question(conn: psycopg.Connection, label: GoldLabel, question: str, deps: RetrievalDeps) -> dict:
    """Reranker scores of the chunks the system would rerank for this question, best first. A reranker
    failure is recorded as service_status "failed" with no scores, never as a low score."""
    (tokens,) = conn.execute("SELECT token_estimate FROM products WHERE product_id = %s",
                             (label.product_id,)).fetchone()
    status = "ok"
    if tokens <= deps.params.full_context_max_tokens:
        strategy = "full_context"
        chunks = conn.execute("SELECT chunk_id, text FROM chunks WHERE product_id = %s ORDER BY chunk_id",
                              (label.product_id,)).fetchall()
        conn.rollback()
        try:
            scored = list(zip([cid for cid, _ in chunks], deps.reranker.score(question, [t for _, t in chunks])))
        except Exception:  # same policy as a failed retrieval channel: record it, keep going
            scored = []
    else:
        strategy = "hybrid"
        result = retrieve(conn, label.product_id, question, embedder=deps.embedder,
                          embedding_model=deps.embedding_model, reranker=deps.reranker, bm25_cache=deps.bm25_cache,
                          params=deps.params, strategy="hybrid")
        conn.rollback()
        status = result.service_status
        scored = [(c.chunk.chunk_id, c.rerank_score) for c in result.pool if c.rerank_score is not None]
    scored.sort(key=lambda pair: -pair[1])
    rank = {cid: i for i, (cid, _) in enumerate(scored, start=1)}
    grade2 = [cid for cid, grade in label.grades.items() if grade == 2]
    grade2_ranks = [rank[cid] for cid in grade2 if cid in rank]
    return {"qid": label.qid, "product_id": label.product_id, "strategy": strategy,
            "service_status": status if scored else "failed", "gold": label.answerability,
            "top1": scored[0][1] if scored else None, "n_scored": len(scored), "grade2_total": len(grade2),
            "grade2_best_rank": min(grade2_ranks) if grade2_ranks else None,
            "top": [[cid, score] for cid, score in scored[:TOP_KEPT]]}


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability that a random positive scores above a random negative; ties count half."""
    if not pos or not neg:
        raise ValueError("AUC needs both classes")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_interval(pos: Sequence[float], neg: Sequence[float], *, seed: int = SEED,
                 resamples: int = RESAMPLES) -> dict | None:
    """AUC with a 95% bootstrap interval. Each class is resampled on its own, so no draw loses a class."""
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    draws = sorted(auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(resamples))
    return {"auc": auc(pos, neg), "lo": _percentile(draws, 0.025), "hi": _percentile(draws, 0.975),
            "n_pos": len(pos), "n_neg": len(neg)}


def best_cut(scores: Sequence[float], answerable: Sequence[bool]) -> tuple[float, float]:
    """(cut, accuracy) of the lowest cut, "answerable when score >= cut", with the best accuracy on these
    same questions; optimistic by construction. math.inf means refusing everything."""
    best = (math.inf, statistics.fmean(not y for y in answerable))
    for cut in sorted(set(scores), reverse=True):
        accuracy = statistics.fmean((s >= cut) == y for s, y in zip(scores, answerable))
        if accuracy >= best[1]:
            best = (cut, accuracy)
    return best


def _auc_of(rows: Sequence[dict]) -> dict | None:
    return auc_interval([r["top1"] for r in rows if r["gold"] != "unanswerable"],
                        [r["top1"] for r in rows if r["gold"] == "unanswerable"])


def summarize(rows: Sequence[dict], workflow: Mapping[str, dict]) -> dict:
    """Metrics over the questions that were scored; `workflow` maps qid to that question's workflow row."""
    scored = [r for r in rows if r["service_status"] != "failed"]
    answerable = [r["gold"] != "unanswerable" for r in scored]
    answered = [bool(workflow[r["qid"]]["answered_binary"]) for r in scored]
    refusals = [r for r, a in zip(scored, answered) if not a]
    false_refusals = [r for r in refusals if r["gold"] != "unanswerable"]
    cut, cut_accuracy = best_cut([r["top1"] for r in scored], answerable)

    def accuracy(predicted: Sequence[bool]) -> float:
        return statistics.fmean(p == y for p, y in zip(predicted, answerable))

    return {
        "questions": len(rows), "scored": len(scored), "failed": len(rows) - len(scored),
        "gold": dict(Counter(r["gold"] for r in scored)),
        "auc_top1": {"all": _auc_of(scored), **{s: _auc_of([r for r in scored if r["strategy"] == s])
                                                for s in STRATEGIES}},
        "top1_median": {"answerable": statistics.median(r["top1"] for r in scored if r["gold"] != "unanswerable"),
                        "unanswerable": statistics.median(r["top1"] for r in scored if r["gold"] == "unanswerable")}
        if any(answerable) and not all(answerable) else None,
        "accuracy": {"workflow": accuracy(answered),
                     "b2_silver": accuracy([bool(workflow[r["qid"]]["answerable_silver"]) for r in scored]),
                     "always_answerable": statistics.fmean(answerable),
                     # None: refusing every question scored best
                     "best_cut_in_sample": {"cut": cut if math.isfinite(cut) else None, "accuracy": cut_accuracy}},
        "refusals": {"n": len(refusals), "false": len(false_refusals), "correct": len(refusals) - len(false_refusals),
                     "false_by_gold": dict(Counter(r["gold"] for r in false_refusals)),
                     "auc_top1": _auc_of(refusals)},
        "false_answers": sum(1 for r, a in zip(scored, answered) if a and r["gold"] == "unanswerable"),
    }


# --- report ------------------------------------------------------------------------------------

def _fmt_auc(a: dict | None) -> str:
    if a is None:
        return "| — | — | — | — |"
    return f"| {a['auc']:.3f} | {a['lo']:.3f}–{a['hi']:.3f} | {a['n_pos']} | {a['n_neg']} |"


def _workflow_label(w: dict) -> str:
    if w.get("route") == "out_of_scope":
        return "超出范围"
    return w.get("status") or ("answered" if w["answered_binary"] else "unanswerable")


def _grade2(r: dict) -> str:
    if r["grade2_total"] == 0:
        return "—"
    return "不在打分范围" if r["grade2_best_rank"] is None else str(r["grade2_best_rank"])


def build_report(config: dict, rows: Sequence[dict], metrics: dict, workflow: Mapping[str, dict]) -> str:
    g, acc, ref = metrics["gold"], metrics["accuracy"], metrics["refusals"]
    cut = acc["best_cut_in_sample"]
    median = metrics["top1_median"]
    lines = [
        f"# reranker 分数检查：{config['name']}", "",
        "reranker 给出的最高分，能不能区分评论能回答和不能回答的题？这次检查用来决定第 6 阶段要不要训练可答性模型。"
        "没有调用 LLM。", "",
        "## 1. 做法", "",
        f"- 题目：金标 {config['split']} 集 {metrics['scored']} 题（answered {g.get('answered', 0)}、"
        f"partial {g.get('partial', 0)}、unanswerable {g.get('unanswerable', 0)}），answered 和 partial 都算可答。"
        f"打分失败 {metrics['failed']} 题，不计入任何指标。",
        f"- 打分范围和系统一致：全量上下文商品给每个块打分；混合检索商品只给重排池里的 "
        f"{config['retrieval_params']['rerank_pool']} 个块打分。",
        "- 查询用题目原文，不是工作流规划后的英文查询；英文题两者基本相同，中文翻译和改写查询不在这次检查里。",
        f"- 工作流和 B2（AmazonQA 银标）的结果取自 `{config['workflow_path']}`，是同一批题此前的金标评测。",
        f"- 区间是 95% bootstrap，可答和不可答分开重采样，{config['resamples']} 次，种子 {config['seed']}。",
        f"- reranker：`{config['rerank_model']}`；git_commit：`{config['git_commit']}`。", "",
        "## 2. 最高分区分可答和不可答（AUC）", "",
        "| 范围 | AUC | 95% 区间 | 可答 | 不可答 |", "|---|---|---|---|---|",
        "| 全部 " + _fmt_auc(metrics["auc_top1"]["all"]),
        "| 混合检索 " + _fmt_auc(metrics["auc_top1"]["hybrid"]),
        "| 全量上下文 " + _fmt_auc(metrics["auc_top1"]["full_context"]), "",
    ]
    if median:
        lines += [f"最高分的中位数：可答 {median['answerable']:.3f}，不可答 {median['unanswerable']:.3f}。", ""]
    cut_text = "全部拒答" if cut["cut"] is None else f"最高分 ≥ {cut['cut']:.4f} 判可答"
    lines += [
        "## 3. 和工作流比可答 / 不可答二分的准确率", "",
        "| 判定方式 | 准确率 |", "|---|---|",
        f"| 工作流 | {acc['workflow']:.3f} |",
        f"| 只看最高分：{cut_text}（阈值在同一批题上挑的，偏乐观） | {cut['accuracy']:.3f} |",
        f"| B2（AmazonQA 银标） | {acc['b2_silver']:.3f} |",
        f"| 全部判可答 | {acc['always_answerable']:.3f} |", "",
        "## 4. 工作流拒答的题", "",
        f"工作流拒答 {ref['n']} 题：误拒 {ref['false']}（金标 "
        + "、".join(f"{k} {v}" for k, v in sorted(ref["false_by_gold"].items())) + f"），正确拒答 {ref['correct']}。"
        f"另有误答 {metrics['false_answers']} 题（金标不可答，工作流作答）。", "",
        "在拒答的题里，最高分区分误拒和正确拒答：", "",
        "| 范围 | AUC | 95% 区间 | 误拒 | 正确拒答 |", "|---|---|---|---|---|",
        "| 工作流拒答的题 " + _fmt_auc(ref["auc_top1"]), "",
        "下表是工作流拒答和误答的题，按最高分从高到低。最后一列是金标 2 级证据块在打分列表里的最好名次；"
        "“—”表示金标没有 2 级块。", "",
        "| qid | 最高分 | 金标 | 工作流 | 检索策略 | 2 级证据最好名次 |", "|---|---|---|---|---|---|",
    ]
    # Refusals plus false answers.
    listed = [r for r in rows if r["service_status"] != "failed"
              and (not workflow[r["qid"]]["answered_binary"] or r["gold"] == "unanswerable")]
    for r in sorted(listed, key=lambda r: -r["top1"]):
        lines.append(f"| {r['qid']} | {r['top1']:.3f} | {r['gold']} | {_workflow_label(workflow[r['qid']])} | "
                     f"{r['strategy']} | {_grade2(r)} |")
    lines += ["", f"这是 {config['split']} 集上的一次检查：不可答只有 {g.get('unanswerable', 0)} 题，区间很宽。", ""]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
    reranker_factory: Callable[[Settings], Reranker] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Check whether the reranker's top score separates answerable "
                                                 "from unanswerable gold questions.")
    parser.add_argument("--labels", type=Path, required=True, help="gold labels.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--workflow", type=Path, required=True,
                        help="per_question.jsonl of a gold eval of the same split (workflow and silver labels)")
    parser.add_argument("--name", required=True, help="output directory name under --out-root")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "reports" / "eval")
    args = parser.parse_args(argv)

    settings = load_settings()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not args.workflow.exists():
        print(f"workflow results not found: {args.workflow}", file=sys.stderr)
        return 2
    workflow = {r["qid"]: r for r in map(json.loads, args.workflow.read_text(encoding="utf-8").splitlines())
                if r.get("system") == "workflow"}
    embedder = (embedder_factory or default_embedder)(settings)
    reranker = (reranker_factory or default_reranker)(settings)
    deps = RetrievalDeps(embedder=embedder, embedding_model=embedder.model_name, reranker=reranker,
                         bm25_cache=BM25Cache(),
                         params=RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens))
    conn = connect(settings.database_url)
    try:
        try:
            labels = sorted((label for label in load_gold(conn, args.labels).values() if label.split == args.split),
                            key=lambda label: label.qid)
        except GoldError as exc:
            print(f"gold labels: {exc}", file=sys.stderr)
            return 2
        if not labels:
            print(f"no labelled questions in {args.split}", file=sys.stderr)
            return 2
        missing = [label.qid for label in labels if label.qid not in workflow]
        if missing:
            print(f"no workflow row in {args.workflow} for: {missing[:10]}", file=sys.stderr)
            return 2
        dataset_id, source_sha256 = conn.execute(
            "SELECT d.dataset_id, d.source_sha256 FROM products p JOIN datasets d USING (dataset_id)"
            " WHERE p.product_id = %s", (labels[0].product_id,)).fetchone()
        rows = []
        for i, label in enumerate(labels, start=1):
            (text,) = conn.execute("SELECT question_text FROM questions WHERE qid = %s", (label.qid,)).fetchone()
            conn.rollback()
            rows.append(score_question(conn, label, text, deps))
            print(f"{i}/{len(labels)} {label.qid} {rows[-1]['strategy']} {rows[-1]['service_status']}",
                  file=sys.stderr, flush=True)
    finally:
        conn.close()

    config = {"name": args.name, "split": args.split, "started_at": started_at, "git_commit": _git_commit(),
              "dataset_id": dataset_id, "source_sha256": source_sha256, "embedding_model": embedder.model_name,
              "rerank_model": reranker.model_name, "retrieval_params": dataclasses.asdict(deps.params),
              "query": "question_text", "seed": SEED, "resamples": RESAMPLES, "questions_in_split": len(labels),
              "labels_path": str(args.labels), "labels_sha256": _sha256(args.labels),
              "workflow_path": str(args.workflow), "workflow_sha256": _sha256(args.workflow)}
    metrics = summarize(rows, workflow)
    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(out_dir / "per_question.jsonl", rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(config, rows, metrics, workflow), encoding="utf-8")
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    raise SystemExit(main())
