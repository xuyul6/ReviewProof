from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from reviewproof.embedding import Embedder
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.rerank import Reranker
from reviewproof.retrieval.types import Candidate, RetrievalParams
from reviewproof.workflow.answer import AnswerClaim, Citation
from reviewproof.workflow.schemas import EvidenceAssessment, GeneratedAnswer


@dataclass(frozen=True)
class RetrievalDeps:
    embedder: Embedder
    embedding_model: str
    reranker: Reranker | None
    bm25_cache: BM25Cache
    params: RetrievalParams = field(default_factory=RetrievalParams)


@dataclass(frozen=True)
class TraceEntry:
    node: str  # plan / retrieve / assess / rewrite / generate / verify
    status: str  # ok / degraded / failed / skipped
    detail: str
    latency_ms: float


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
    generated: GeneratedAnswer
    claims: tuple[AnswerClaim, ...]
    citations: tuple[Citation, ...]
    unknowns: tuple[str, ...]
    status: str | None
    verdict: str | None
    answer_text: str
    service_failed: bool
    degraded: bool
    retrieval_modes: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
    usage: Annotated[list[tuple[int, int]], operator.add]
