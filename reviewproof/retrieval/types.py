from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    review_id: str
    product_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    """One result of a single channel; its rank is its position in the list + 1."""

    chunk: ChunkRow
    score: float


@dataclass(frozen=True)
class RetrievalParams:
    rrf_k: int = 60
    channel_top_k: int = 50
    rerank_pool: int = 40
    evidence_k: int = 8
    full_context_max_tokens: int = 6000


@dataclass(frozen=True)
class Candidate:
    chunk: ChunkRow
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class ChannelStatus:
    name: str  # "dense" | "bm25" | "rerank"
    status: str  # "ok" | "failed" | "skipped"
    reason: str | None
    latency_ms: float
    hits: int


@dataclass(frozen=True)
class RetrievalResult:
    product_id: str
    query: str
    strategy: str  # "full_context" | "hybrid"
    mode: str  # channels that succeeded, e.g. "dense+bm25+rerank"; "full_context"; "none" if no channel did
    # "failed": a recall channel failed and nothing usable came back; never read it as "no evidence"
    service_status: str  # "ok" | "degraded" | "failed"
    pool: tuple[Candidate, ...]  # top rerank_pool in RRF order (carrying rerank_score if reranked), for Recall@K
    candidates: tuple[Candidate, ...]  # final evidence, at most evidence_k
    channels: tuple[ChannelStatus, ...]
