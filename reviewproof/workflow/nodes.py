"""Workflow nodes. Each returns a partial state update and appends one trace entry.

Degradation stays inside the nodes: a failed plan falls back to a heuristic; a failed retrieval,
assessment or generation marks the run as a service failure, which is never reported as
"the reviews don't confirm this".
"""
from __future__ import annotations

import re
import time

import psycopg

from reviewproof.llm import LLM, LLMError
from reviewproof.retrieval import retrieve
from reviewproof.workflow.answer import final_status, render_answer, verify_claims
from reviewproof.workflow.prompts import (ASSESS_SYSTEM, GENERATE_SYSTEM, PLAN_SYSTEM, assess_prompt,
                                          generate_prompt, plan_prompt)
from reviewproof.workflow.schemas import EvidenceAssessment, GeneratedAnswer, QueryPlan
from reviewproof.workflow.state import QAState, RetrievalDeps, TraceEntry

_YES_NO_EN = re.compile(r"^\s*(is|are|does|do|did|can|will|would|should|could|has|have)\b", re.IGNORECASE)
_YES_NO_ZH = ("吗", "是否", "能不能", "会不会")


def looks_yes_no(question: str) -> bool:
    return bool(_YES_NO_EN.match(question)) or any(marker in question for marker in _YES_NO_ZH)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class WorkflowNodes:
    def __init__(self, conn: psycopg.Connection, llm: LLM, retrieval: RetrievalDeps, max_rewrites: int):
        self.conn = conn
        self.llm = llm
        self.retrieval = retrieval
        self.max_rewrites = max_rewrites

    # --- nodes -------------------------------------------------------------------------------

    def plan(self, state: QAState) -> dict:
        start = time.perf_counter()
        question = state["question"]
        try:
            result = self.llm.structured(system=PLAN_SYSTEM, prompt=plan_prompt(question), schema=QueryPlan)
        except LLMError as exc:
            return {"route": "review_question", "out_of_scope_reason": "", "is_yes_no": looks_yes_no(question),
                    "search_queries": [question], "degraded": True,
                    "trace": [TraceEntry("plan", "degraded", f"heuristic plan after LLMError: {exc}", _ms(start))]}
        p = result.output
        detail = f"route={p.route} yes_no={p.is_yes_no} search_query={p.search_query!r}"
        return {"route": p.route, "out_of_scope_reason": p.out_of_scope_reason, "is_yes_no": p.is_yes_no,
                "search_queries": [p.search_query.strip() or question],
                "usage": [(result.input_tokens, result.output_tokens)],
                "trace": [TraceEntry("plan", "ok", detail, _ms(start))]}

    def retrieve(self, state: QAState) -> dict:
        start = time.perf_counter()
        result, error = self._retrieve(state, state["search_queries"][0])
        if error:
            return {"service_failed": True, "trace": [TraceEntry("retrieve", "failed", error, _ms(start))]}
        status = "degraded" if result.service_status == "degraded" else "ok"
        return {"candidates": list(result.candidates), "strategy": result.strategy,
                "retrieval_modes": [result.mode], "degraded": state.get("degraded", False) or status == "degraded",
                "trace": [TraceEntry("retrieve", status,
                                     f"{result.strategy} mode={result.mode} candidates={len(result.candidates)}",
                                     _ms(start))]}

    def assess(self, state: QAState) -> dict:
        start = time.perf_counter()
        candidates = state["candidates"]
        if not candidates:
            assessment = EvidenceAssessment(status="unanswerable", supporting_chunk_ids=[],
                                            missing=[state["question"]], rewrite_query="")
            return {"assessment": assessment,
                    "trace": [TraceEntry("assess", "ok", "no candidates: unanswerable", _ms(start))]}
        try:
            result = self.llm.structured(system=ASSESS_SYSTEM, prompt=assess_prompt(state["question"], candidates),
                                         schema=EvidenceAssessment)
        except LLMError as exc:
            return {"service_failed": True, "trace": [TraceEntry("assess", "failed", f"LLMError: {exc}", _ms(start))]}
        known = {c.chunk.chunk_id for c in candidates}
        a = result.output
        ignored = [i for i in a.supporting_chunk_ids if i not in known]
        assessment = a.model_copy(update={"supporting_chunk_ids": [i for i in a.supporting_chunk_ids if i in known]})
        detail = f"status={a.status} supporting={len(assessment.supporting_chunk_ids)} missing={a.missing}"
        if ignored:
            detail += f" ignored_ids={ignored}"
        return {"assessment": assessment, "usage": [(result.input_tokens, result.output_tokens)],
                "trace": [TraceEntry("assess", "ok", detail, _ms(start))]}

    def rewrite(self, state: QAState) -> dict:
        start = time.perf_counter()
        query = state["assessment"].rewrite_query.strip()
        update: dict = {"rewrites_used": state.get("rewrites_used", 0) + 1,
                        "search_queries": [*state["search_queries"], query], "rewrite_found_new": False}
        result, error = self._retrieve(state, query)
        if error:
            update.update(degraded=True, trace=[TraceEntry(
                "rewrite", "degraded", f"query={query!r}; retrieval failed, keeping candidates: {error}", _ms(start))])
            return update
        known = {c.chunk.chunk_id for c in state["candidates"]}
        new = [c for c in result.candidates if c.chunk.chunk_id not in known]
        update["retrieval_modes"] = [result.mode]
        if not new:
            update["trace"] = [TraceEntry("rewrite", "ok", f"query={query!r}; no_new_evidence", _ms(start))]
            return update
        limit = 2 * self.retrieval.params.evidence_k
        update.update(candidates=(state["candidates"] + new)[:limit], rewrite_found_new=True,
                      trace=[TraceEntry("rewrite", "ok", f"query={query!r}; new_candidates={len(new)}", _ms(start))])
        return update

    def generate(self, state: QAState) -> dict:
        start = time.perf_counter()
        a = state["assessment"]
        prompt = generate_prompt(state["question"], state["is_yes_no"], a.status, state["candidates"], a.missing)
        try:
            result = self.llm.structured(system=GENERATE_SYSTEM, prompt=prompt, schema=GeneratedAnswer)
        except LLMError as exc:
            return {"service_failed": True,
                    "trace": [TraceEntry("generate", "failed", f"LLMError: {exc}", _ms(start))]}
        return {"generated": result.output, "usage": [(result.input_tokens, result.output_tokens)],
                "trace": [TraceEntry("generate", "ok", f"claims={len(result.output.claims)}", _ms(start))]}

    def verify(self, state: QAState) -> dict:
        start = time.perf_counter()
        g = state["generated"]
        claims, citations, dropped = verify_claims(g.claims, state["candidates"], state["product_id"])
        unknowns = tuple(dict.fromkeys([*g.unknowns, *state["assessment"].missing]))
        status = final_status(state["assessment"].status, claims, unknowns)
        detail = f"claims_kept={len(claims)}/{len(g.claims)} status={status}"
        if dropped:
            detail += f" dropped_ids={list(dropped)}"
        return {"claims": claims, "citations": citations, "unknowns": unknowns, "status": status,
                "trace": [TraceEntry("verify", "ok", detail, _ms(start))]}

    def finalize(self, state: QAState) -> dict:
        failed = state.get("service_failed", False)
        route = state.get("route", "review_question")
        if failed:
            status, verdict = None, None
        elif route == "out_of_scope":
            status, verdict = "unanswerable", None
        else:
            status = state.get("status") or state["assessment"].status
            if not state.get("is_yes_no"):
                verdict = None
            elif status == "unanswerable" or "generated" not in state:
                verdict = "unknown"
            else:
                verdict = state["generated"].verdict or "unknown"
        claims = state.get("claims", ()) if status in ("answered", "partial") else ()
        unknowns = state.get("unknowns", ()) if status in ("answered", "partial") else ()
        service_status = "failed" if failed else "degraded" if state.get("degraded") else "ok"
        text = render_answer(question=state["question"], route=route,
                             out_of_scope_reason=state.get("out_of_scope_reason", ""), service_status=service_status,
                             status=status, verdict=verdict, claims=claims, unknowns=unknowns)
        return {"status": status, "verdict": verdict, "claims": claims, "unknowns": unknowns,
                "citations": state.get("citations", ()) if claims else (), "answer_text": text}

    # --- routing -----------------------------------------------------------------------------

    def after_plan(self, state: QAState) -> str:
        return "finalize" if state.get("route") == "out_of_scope" else "retrieve"

    def after_retrieve(self, state: QAState) -> str:
        return "finalize" if state.get("service_failed") else "assess"

    def after_assess(self, state: QAState) -> str:
        if state.get("service_failed"):
            return "finalize"
        a = state["assessment"]
        query = a.rewrite_query.strip()
        if (a.status in ("partial", "unanswerable") and state.get("rewrites_used", 0) < self.max_rewrites
                and state.get("strategy") == "hybrid" and query and query not in state["search_queries"]):
            return "rewrite"
        return self._answer_or_finish(state)

    def after_rewrite(self, state: QAState) -> str:
        return "assess" if state.get("rewrite_found_new") else self._answer_or_finish(state)

    def after_generate(self, state: QAState) -> str:
        return "finalize" if state.get("service_failed") else "verify"

    # --- helpers -----------------------------------------------------------------------------

    def _answer_or_finish(self, state: QAState) -> str:
        return "generate" if state["assessment"].status in ("answered", "partial") else "finalize"

    def _retrieve(self, state: QAState, query: str):
        deps = self.retrieval
        try:
            result = retrieve(self.conn, state["product_id"], query, embedder=deps.embedder,
                              embedding_model=deps.embedding_model, reranker=deps.reranker,
                              bm25_cache=deps.bm25_cache, params=deps.params, strategy="auto")
        except Exception as exc:  # database errors outside the channels
            self.conn.rollback()
            return None, f"{type(exc).__name__}: {exc}"
        if result.service_status == "failed":
            reasons = "; ".join(f"{c.name}: {c.reason}" for c in result.channels if c.status == "failed")
            return None, f"retrieval failed ({reasons})"
        return result, None
