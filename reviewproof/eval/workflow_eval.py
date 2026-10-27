"""Run the workflow and the B0 / B1 baselines over sampled questions, one JSON line per
(question, system), appended and flushed as it goes so an interrupted run resumes without
paying twice. Only question_text (and, for B0, the category) ever reaches a prompt."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Sequence

import psycopg

from reviewproof.eval.baselines import b0_no_evidence
from reviewproof.eval.dataset import EvalQuestion
from reviewproof.llm import LLM, LLMError
from reviewproof.workflow.graph import answer_question
from reviewproof.workflow.state import RetrievalDeps

SYSTEMS = ("workflow", "b0_no_evidence", "b1_full_context")

# Claude Haiku 5.5, USD per million tokens, as of 2026-10-27.
PRICE_SHORT = (0.10, 0.50)  # prompts up to 100k tokens
PRICE_LONG = (0.50, 2.50)   # prompts over 100k tokens
LONG_PROMPT_TOKENS = 100_000


def cost_usd(input_tokens: int, output_tokens: int, llm_calls: int) -> float:
    """Approximate: per-call input size is unknown, so the tier is chosen from the average per call."""
    if llm_calls <= 0:
        return 0.0
    rate_in, rate_out = PRICE_LONG if input_tokens / llm_calls > LONG_PROMPT_TOKENS else PRICE_SHORT
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def _read_done(out_path: Path) -> list[dict]:
    """Complete rows already written; a half-written last line (interrupted run) is dropped."""
    if not out_path.exists():
        return []
    rows = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def _base_row(q: EvalQuestion, system: str) -> dict:
    return {"qid": q.qid, "system": system, "product_id": q.product_id, "category": q.category,
            "question_type": q.question_type, "strategy": q.strategy, "answerable_silver": q.answerable_silver,
            "status": None, "service_status": "failed", "route": None, "verdict": None, "answered_binary": None,
            "n_citations": 0, "citation_out_of_product": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "latency_ms": 0.0, "rewrites_used": 0, "retrieval_modes": [], "answer_text": "",
            "error": None}


def _run_one(conn, q: EvalQuestion, system: str, llm: LLM, deps: RetrievalDeps, max_rewrites: int) -> dict:
    row = _base_row(q, system)
    start = time.perf_counter()
    try:
        if system == "b0_no_evidence":
            try:
                result = b0_no_evidence(llm, q.question_text, q.category)
            except LLMError as exc:
                row["error"] = f"LLMError: {exc}"
            else:
                row.update(service_status="ok", status="answered" if result.output.answered else "unanswerable",
                           answered_binary=result.output.answered, llm_calls=1, input_tokens=result.input_tokens,
                           output_tokens=result.output_tokens, answer_text=result.output.answer)
        else:
            run_deps = deps if system == "workflow" else dataclasses.replace(
                deps, params=dataclasses.replace(deps.params, full_context_max_tokens=10**9))
            r = answer_question(conn, q.product_id, q.question_text, llm=llm, retrieval=run_deps,
                                max_rewrites=max_rewrites)
            failed = r.service_status == "failed"
            row.update(status=r.status, service_status=r.service_status, route=r.route, verdict=r.verdict,
                       answered_binary=None if failed else r.status in ("answered", "partial"),
                       n_citations=len(r.citations),
                       citation_out_of_product=sum(not c.chunk_id.startswith(f"{q.product_id}-r") for c in r.citations),
                       llm_calls=r.llm_calls, input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                       rewrites_used=r.rewrites_used, retrieval_modes=list(r.retrieval_modes),
                       answer_text=r.answer_text)
    except Exception as exc:  # keep going: one broken question must not stop the evaluation
        conn.rollback()
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_ms"] = (time.perf_counter() - start) * 1000
    row["cost_usd"] = cost_usd(row["input_tokens"], row["output_tokens"], row["llm_calls"])
    return row


def run_workflow_eval(conn: psycopg.Connection, questions: Sequence[EvalQuestion], *, llm: LLM,
                      deps: RetrievalDeps, max_rewrites: int, systems: Sequence[str], out_path: Path,
                      progress=None) -> list[dict]:
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    rows = _read_done(out_path)
    done = {(r.get("qid"), r.get("system")) for r in rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out:
        for q in questions:
            for system in systems:
                if system == "b1_full_context" and q.strategy != "hybrid":
                    continue  # identical to the workflow on full-context products
                if (q.qid, system) in done:
                    continue
                row = _run_one(conn, q, system, llm, deps, max_rewrites)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                if progress:
                    progress(row)
    return rows
