"""The QA workflow as a LangGraph state graph, the only orchestration path.

plan → (out of scope: finalize) → retrieve → assess → [rewrite → assess] → generate → verify → finalize
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import psycopg
from langgraph.graph import END, START, StateGraph

from reviewproof.llm import LLM
from reviewproof.retrieval import ProductNotFoundError
from reviewproof.workflow.answer import AnswerClaim, Citation
from reviewproof.workflow.nodes import WorkflowNodes
from reviewproof.workflow.state import QAState, RetrievalDeps, TraceEntry


@dataclass(frozen=True)
class QAResult:
    product_id: str
    question: str
    route: str  # review_question / out_of_scope
    status: str | None  # answered / partial / unanswerable; None on service failure
    service_status: str  # ok / degraded / failed
    verdict: str | None  # yes / no / mixed / unknown, only for yes/no questions
    claims: tuple[AnswerClaim, ...]
    unknowns: tuple[str, ...]
    citations: tuple[Citation, ...]
    answer_text: str
    search_queries: tuple[str, ...]
    retrieval_modes: tuple[str, ...]
    rewrites_used: int
    trace: tuple[TraceEntry, ...]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float


def build_graph(nodes: WorkflowNodes):
    graph = StateGraph(QAState)
    for name in ("plan", "retrieve", "assess", "rewrite", "generate", "verify", "finalize"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", nodes.after_plan, ["retrieve", "finalize"])
    graph.add_conditional_edges("retrieve", nodes.after_retrieve, ["assess", "finalize"])
    graph.add_conditional_edges("assess", nodes.after_assess, ["rewrite", "generate", "finalize"])
    graph.add_conditional_edges("rewrite", nodes.after_rewrite, ["assess", "generate", "finalize"])
    graph.add_conditional_edges("generate", nodes.after_generate, ["verify", "finalize"])
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def stream_answer(conn: psycopg.Connection, product_id: str, question: str, *, llm: LLM,
                  retrieval: RetrievalDeps, max_rewrites: int = 1) -> Iterator[TraceEntry | QAResult]:
    """Yield each node's trace entry as it finishes, then the QAResult. Input is checked on the
    first next(): an empty question raises ValueError, an unknown product ProductNotFoundError."""
    if not question.strip():
        raise ValueError("question is empty")
    exists = conn.execute("SELECT 1 FROM products WHERE product_id = %s", (product_id,)).fetchone()
    conn.rollback()
    if exists is None:
        raise ProductNotFoundError(product_id)

    start = time.perf_counter()
    graph = build_graph(WorkflowNodes(conn, llm, retrieval, max_rewrites))
    state: dict = {}
    initial = {"product_id": product_id, "question": question, "rewrites_used": 0}
    for mode, chunk in graph.stream(initial, stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
            continue
        for update in chunk.values():
            yield from (update or {}).get("trace", [])

    usage = state.get("usage", [])
    service_status = ("failed" if state.get("service_failed")
                      else "degraded" if state.get("degraded") else "ok")
    yield QAResult(
        product_id=product_id, question=question, route=state.get("route", "review_question"),
        status=state["status"], service_status=service_status, verdict=state["verdict"],
        claims=tuple(state["claims"]), unknowns=tuple(state["unknowns"]), citations=tuple(state["citations"]),
        answer_text=state["answer_text"], search_queries=tuple(state.get("search_queries", ())),
        retrieval_modes=tuple(state.get("retrieval_modes", ())), rewrites_used=state.get("rewrites_used", 0),
        trace=tuple(state.get("trace", ())), llm_calls=len(usage),
        input_tokens=sum(u[0] for u in usage), output_tokens=sum(u[1] for u in usage),
        latency_ms=(time.perf_counter() - start) * 1000)


def answer_question(conn: psycopg.Connection, product_id: str, question: str, *, llm: LLM,
                    retrieval: RetrievalDeps, max_rewrites: int = 1) -> QAResult:
    *_, result = stream_answer(conn, product_id, question, llm=llm, retrieval=retrieval, max_rewrites=max_rewrites)
    return result
