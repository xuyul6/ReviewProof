"""Workflow nodes. Each returns a partial state update and appends one trace entry.

Degradation stays inside the nodes: a failed plan falls back to a heuristic; a failed retrieval,
assessment or generation marks the run as a service failure, which is never reported as
"the reviews don't confirm this". A failed general note only drops the note.
"""
from __future__ import annotations

import re
import time

import psycopg
from langgraph.graph import END

from reviewproof.llm import LLM, LLMError
from reviewproof.retrieval import retrieve
from reviewproof.workflow.answer import final_status, render_answer, render_note, verify_claims
from reviewproof.workflow.prompts import (ADVISE_SYSTEM, ASSESS_SYSTEM, GENERATE_SYSTEM, PLAN_SYSTEM, advise_prompt,
                                          assess_prompt, generate_prompt, plan_prompt)
from reviewproof.workflow.schemas import (EvidenceAssessment, GeneralNote, GeneratedAnswer, GeneratedYesNoAnswer,
                                          QueryPlan)
from reviewproof.workflow.state import QAState, RetrievalDeps, RetrievalSnapshot, TraceEntry

_YES_NO_EN = re.compile(r"^\s*(is|are|does|do|did|can|will|would|should|could|has|have)\b", re.IGNORECASE)
_YES_NO_ZH = ("吗", "是否", "能不能", "会不会")


def looks_yes_no(question: str) -> bool:
    return bool(_YES_NO_EN.match(question)) or any(marker in question for marker in _YES_NO_ZH)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class WorkflowNodes:
    def __init__(self, conn: psycopg.Connection, llm: LLM, retrieval: RetrievalDeps, max_rewrites: int,
                 advise: bool = False):
        self.conn = conn
        self.llm = llm
        self.retrieval = retrieval
        self.max_rewrites = max_rewrites
        self.advise_enabled = advise

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
        query = state["search_queries"][0]
        result, error = self._retrieve(state, query)
        if error:
            return {"service_failed": True, "trace": [TraceEntry("retrieve", "failed", error, _ms(start))]}
        status = "degraded" if result.service_status == "degraded" else "ok"
        return {"candidates": list(result.candidates), "strategy": result.strategy,
                "retrieval_modes": [result.mode], "retrieval_snapshots": [RetrievalSnapshot.of(query, result)],
                "degraded": state.get("degraded", False) or status == "degraded",
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
        # Each rewrite may add up to evidence_k chunks, so a second rewrite can still add evidence.
        limit = self.retrieval.params.evidence_k * (update["rewrites_used"] + 1)
        merged = (state["candidates"] + new)[:limit]
        added = len(merged) - len(state["candidates"])
        status = "degraded" if result.service_status == "degraded" else "ok"
        update["retrieval_modes"] = [result.mode]
        update["retrieval_snapshots"] = [RetrievalSnapshot.of(query, result)]
        if status == "degraded":
            update["degraded"] = True
        if added <= 0:
            update["trace"] = [TraceEntry("rewrite", status, f"query={query!r}; no_new_evidence", _ms(start))]
            return update
        update.update(candidates=merged, rewrite_found_new=True,
                      trace=[TraceEntry("rewrite", status, f"query={query!r}; new_candidates={added}", _ms(start))])
        return update

    def generate(self, state: QAState) -> dict:
        start = time.perf_counter()
        a = state["assessment"]
        prompt = generate_prompt(state["question"], state["is_yes_no"], a.status, state["candidates"], a.missing)
        try:
            schema = GeneratedYesNoAnswer if state["is_yes_no"] else GeneratedAnswer
            result = self.llm.structured(system=GENERATE_SYSTEM, prompt=prompt, schema=schema)
        except LLMError as exc:
            return {"service_failed": True,
                    "trace": [TraceEntry("generate", "failed", f"LLMError: {exc}", _ms(start))]}
        g = result.output
        detail = f"claims={len(g.claims)}"
        if isinstance(g, GeneratedYesNoAnswer):
            detail += f" verdict_question={g.verdict_question!r} verdict={g.verdict}"
        return {"generated": g, "usage": [(result.input_tokens, result.output_tokens)],
                "trace": [TraceEntry("generate", "ok", detail, _ms(start))]}

    def verify(self, state: QAState) -> dict:
        start = time.perf_counter()
        g = state["generated"]
        claims, citations, dropped = verify_claims(g.claims, state["candidates"], state["product_id"])
        # Gaps come from the assessment: the generator tends to list caveats, which must not
        # turn an answered question into a partial one.
        unknowns = tuple(dict.fromkeys(state["assessment"].missing))
        status = final_status(state["assessment"].status, claims, g.verdict)
        detail = f"claims_kept={len(claims)}/{len(g.claims)} status={status}"
        if status != state["assessment"].status:
            detail += f" downgraded_from={state['assessment'].status}"
        if dropped:
            detail += f" dropped_ids={list(dropped)}"
        if g.unknowns:
            detail += f" generator_unknowns={g.unknowns}"
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

    def advise(self, state: QAState) -> dict:
        """General knowledge or where to check, apart from the answer. Sees no review text, and its
        failure drops the note without touching the run's status or service status."""
        start = time.perf_counter()
        out_of_scope = state.get("route") == "out_of_scope"
        try:
            title, category = self.conn.execute("SELECT title, category FROM products WHERE product_id = %s",
                                                (state["product_id"],)).fetchone()
            self.conn.rollback()
            prompt = advise_prompt(state["question"], title=title, category=category,
                                   missing=state["unknowns"] if state["status"] == "partial" else (),
                                   out_of_scope_reason=state.get("out_of_scope_reason", "") if out_of_scope else "")
            result = self.llm.structured(system=ADVISE_SYSTEM, prompt=prompt, schema=GeneralNote)
        except (LLMError, psycopg.Error) as exc:
            self.conn.rollback()
            return {"trace": [TraceEntry("advise", "failed", f"{type(exc).__name__}: {exc}; note omitted", _ms(start))]}
        n = result.output
        update: dict = {"usage": [(result.input_tokens, result.output_tokens)]}
        if not n.text.strip():
            update["trace"] = [TraceEntry("advise", "ok", "empty note dropped", _ms(start))]
            return update
        update.update(note=render_note(state["question"], n.kind, n.text),
                      trace=[TraceEntry("advise", "ok", f"kind={n.kind}", _ms(start))])
        return update

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

    def after_finalize(self, state: QAState) -> str:
        if not self.advise_enabled or state.get("service_failed"):
            return END
        status = state["status"]
        if (state.get("route") == "out_of_scope" or status == "unanswerable"
                or (status == "partial" and state["unknowns"])):
            return "advise"
        return END

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
