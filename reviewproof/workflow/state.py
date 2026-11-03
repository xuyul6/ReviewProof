from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from reviewproof.embedding import Embedder
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.rerank import Reranker
from reviewproof.retrieval.types import Candidate, ChannelStatus, RetrievalParams, RetrievalResult
from reviewproof.workflow.answer import AnswerClaim, Citation, Note
from reviewproof.workflow.schemas import EvidenceAssessment, GeneratedAnswer, GeneratedYesNoAnswer


@dataclass(frozen=True)
class RetrievalDeps:
    embedder: Embedder
    embedding_model: str
    reranker: Reranker | None
    bm25_cache: BM25Cache
    params: RetrievalParams = field(default_factory=RetrievalParams)


@dataclass(frozen=True)
class TraceEntry:
    node: str  # plan / retrieve / assess / rewrite / generate / verify / advise
    status: str  # ok / degraded / failed / skipped
    detail: str
    latency_ms: float


@dataclass(frozen=True)
class RetrievalSnapshot:
    """One retrieval's final evidence with every channel's rank and score (no review text)."""

    query: str
    strategy: str
    mode: str
    service_status: str
    channels: tuple[ChannelStatus, ...]
    candidates: tuple[dict, ...]

    @classmethod
    def of(cls, query: str, result: RetrievalResult) -> "RetrievalSnapshot":
        candidates = tuple(
            {"chunk_id": c.chunk.chunk_id, "review_id": c.chunk.review_id, "dense_rank": c.dense_rank,
             "dense_score": c.dense_score, "bm25_rank": c.bm25_rank, "bm25_score": c.bm25_score,
             "rrf_score": c.rrf_score, "rerank_score": c.rerank_score}
            for c in result.candidates)
        return cls(query, result.strategy, result.mode, result.service_status, result.channels, candidates)


class QAState(TypedDict, total=False):
    product_id: str
    question: str
    route: str
    out_of_scope_reason: str
    is_yes_no: bool
    search_queries: list[str]
    strategy: str
    candidates: list[Candidate]
    assessment: EvidenceAssessment
    rewrites_used: int
    rewrite_found_new: bool
    generated: GeneratedAnswer | GeneratedYesNoAnswer
    claims: tuple[AnswerClaim, ...]
    citations: tuple[Citation, ...]
    unknowns: tuple[str, ...]
    status: str | None
    verdict: str | None
    answer_text: str
    note: Note
    service_failed: bool
    degraded: bool
    retrieval_modes: Annotated[list[str], operator.add]
    retrieval_snapshots: Annotated[list[RetrievalSnapshot], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
    usage: Annotated[list[tuple[int, int]], operator.add]
