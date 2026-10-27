"""python -m reviewproof.eval --split dev --name NAME [--sample 50] [--retrieval-only] [--systems ...]

Retrieval is evaluated on the whole split (no API cost). The workflow and baselines run on a
stratified sample and resume from per_question.jsonl, so re-running never pays twice.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from reviewproof.config import Settings, load_settings
from reviewproof.db import connect
from reviewproof.embedding import Embedder
from reviewproof.eval.alignment import silver_qrels
from reviewproof.eval.dataset import load_questions, sample_questions
from reviewproof.eval.metrics import SEED
from reviewproof.eval.report import build_report
from reviewproof.eval.retrieval_eval import evaluate_retrieval
from reviewproof.eval.workflow_eval import SYSTEMS, run_workflow_eval
from reviewproof.llm import LLM
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.cli import default_embedder, default_reranker
from reviewproof.retrieval.rerank import Reranker
from reviewproof.workflow.cli import default_llm
from reviewproof.workflow.state import RetrievalDeps

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "reviewproof"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    llm_factory: Callable[[Settings], LLM] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
    reranker_factory: Callable[[Settings], Reranker] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answerability against AmazonQA silver labels.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--name", required=True, help="output directory name under --out-root")
    parser.add_argument("--sample", type=int, default=50, help="questions for the workflow and baselines")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "reports" / "eval")
    args = parser.parse_args(argv)
    systems = tuple(s for s in args.systems.split(",") if s)
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        print(f"unknown systems: {sorted(unknown)}; choose from {SYSTEMS}", file=sys.stderr)
        return 2

    settings = load_settings()
    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = (embedder_factory or default_embedder)(settings)
    deps = RetrievalDeps(embedder=embedder, embedding_model=embedder.model_name,
                         reranker=(reranker_factory or default_reranker)(settings), bm25_cache=BM25Cache(),
                         params=RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens))
    conn = connect(settings.database_url)
    try:
        datasets = conn.execute("SELECT dataset_id, source_sha256 FROM datasets ORDER BY dataset_id").fetchall()
        conn.rollback()
        questions = load_questions(conn, args.split, settings.full_context_max_tokens)
        sample = [] if args.retrieval_only else sample_questions(questions, args.sample, args.seed)
        llm = None if args.retrieval_only else (llm_factory or default_llm)(settings)
        config = {
            "name": args.name, "split": args.split, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "dataset_id": ",".join(d[0] for d in datasets),
            "source_sha256": ",".join(d[1] for d in datasets), "embedding_model": embedder.model_name,
            "rerank_model": getattr(deps.reranker, "model_name", None),
            "llm_model": getattr(llm, "model_name", None), "retrieval_params": dataclasses.asdict(deps.params),
            "max_rewrites": settings.max_rewrites, "seed": args.seed, "questions_in_split": len(questions),
            "sample": len(sample), "systems": list(systems) if sample else [],
            "cost_note": "Claude Haiku 5.5 prices as of 2026-10-27; each call priced at its own tier (over 100k input tokens is long-context)",
        }
        (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"retrieval: {len(questions)} questions in {args.split}", file=sys.stderr, flush=True)
        retrieval_rows = evaluate_retrieval(conn, questions, silver_qrels(conn, questions), deps)
        _write_jsonl(out_dir / "retrieval.jsonl", retrieval_rows)

        workflow_rows = None
        if sample:
            print(f"workflow: {len(sample)} questions × {systems}", file=sys.stderr, flush=True)
            workflow_rows = run_workflow_eval(
                conn, sample, llm=llm, deps=deps, max_rewrites=settings.max_rewrites, systems=systems,
                out_path=out_dir / "per_question.jsonl",
                progress=lambda r: print(f"  {r['qid']} {r['system']}: {r['status']} / {r['service_status']} "
                                         f"{r['latency_ms'] / 1000:.1f}s", file=sys.stderr, flush=True))
            sampled = {q.qid for q in sample}
            workflow_rows = [r for r in workflow_rows if r["qid"] in sampled and r["system"] in systems]
    finally:
        conn.close()

    build_report(retrieval_rows=retrieval_rows, workflow_rows=workflow_rows, config=config, out_dir=out_dir)
    print(out_dir / "report.md")
    return 0
