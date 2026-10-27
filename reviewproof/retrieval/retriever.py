"""Pick a strategy for one product and question, run the channels, fuse, rerank and report.

Every channel failure is caught and recorded instead of raised, so the caller always learns
which channels ran. service_status "failed" means nothing could be searched; it must never be
read as "the reviews do not answer this".
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Mapping, Sequence

import psycopg

from reviewproof.embedding import Embedder
from reviewproof.retrieval.bm25 import BM25Cache, load_product_chunks, tokenize
from reviewproof.retrieval.dense import dense_search
from reviewproof.retrieval.rerank import Reranker
from reviewproof.retrieval.types import Candidate, ChannelStatus, Hit, RetrievalParams, RetrievalResult

STRATEGIES = ("auto", "hybrid", "full_context")
RECALL_CHANNELS = ("dense", "bm25")


class ProductNotFoundError(LookupError):
    pass


def rrf_fuse(rankings: Mapping[str, Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_ids in rankings.values():
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def retrieve(
    conn: psycopg.Connection,
    product_id: str,
    query: str,
    *,
    embedder: Embedder,
    embedding_model: str,
    reranker: Reranker | None,
    bm25_cache: BM25Cache,
    params: RetrievalParams = RetrievalParams(),
    strategy: str = "auto",
) -> RetrievalResult:
    if not query.strip():
        raise ValueError("query is empty")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    row = conn.execute("SELECT token_estimate FROM products WHERE product_id = %s", (product_id,)).fetchone()
    conn.rollback()
    if row is None:
        raise ProductNotFoundError(product_id)

    if strategy == "full_context" or (strategy == "auto" and row[0] <= params.full_context_max_tokens):
        result = _full_context(conn, product_id, query)
    else:
        result = _hybrid(conn, product_id, query, embedder, embedding_model, reranker, bm25_cache, params)
    _check_scope(result)
    return result


def _full_context(conn: psycopg.Connection, product_id: str, query: str) -> RetrievalResult:
    candidates = tuple(Candidate(chunk) for chunk in load_product_chunks(conn, product_id))
    channels = tuple(ChannelStatus(name, "skipped", "full_context", 0.0, 0) for name in (*RECALL_CHANNELS, "rerank"))
    return RetrievalResult(product_id, query, "full_context", "full_context", "ok", candidates, candidates, channels)


def _hybrid(
    conn: psycopg.Connection,
    product_id: str,
    query: str,
    embedder: Embedder,
    embedding_model: str,
    reranker: Reranker | None,
    bm25_cache: BM25Cache,
    params: RetrievalParams,
) -> RetrievalResult:
    hits: dict[str, list[Hit]] = {}
    channels = [
        _run_channel(conn, "dense", hits, lambda: dense_search(
            conn, product_id, embedder.embed_query(query), embedding_model, params.channel_top_k)),
        _run_channel(conn, "bm25", hits, lambda: bm25_cache.search(conn, product_id, query, params.channel_top_k))
        if tokenize(query) else ChannelStatus("bm25", "skipped", "no_query_terms", 0.0, 0),
    ]

    pool = _fuse(hits, params)
    candidates: tuple[Candidate, ...]
    if reranker is None:
        channels.append(ChannelStatus("rerank", "skipped", "disabled", 0.0, 0))
        candidates = pool[: params.evidence_k]
    elif not pool:
        channels.append(ChannelStatus("rerank", "skipped", "no_candidates", 0.0, 0))
        candidates = ()
    else:
        start = time.perf_counter()
        try:
            scores = reranker.score(query, [c.chunk.text for c in pool])
            pool = tuple(replace(c, rerank_score=s) for c, s in zip(pool, scores))
            order = sorted(range(len(pool)), key=lambda i: (-pool[i].rerank_score, i))
            candidates = tuple(pool[i] for i in order[: params.evidence_k])
            channels.append(ChannelStatus("rerank", "ok", None, _ms(start), len(candidates)))
        except Exception as exc:  # degrade to RRF order
            channels.append(ChannelStatus("rerank", "failed", f"{type(exc).__name__}: {exc}", _ms(start), 0))
            candidates = pool[: params.evidence_k]

    recall = [c for c in channels if c.name in RECALL_CHANNELS]
    if not any(c.status == "ok" for c in recall) and any(c.status == "failed" for c in recall):
        service_status = "failed"
    elif any(c.status == "failed" for c in channels):
        service_status = "degraded"
    else:
        service_status = "ok"
    mode = "+".join(c.name for c in channels if c.status == "ok") if any(c.status == "ok" for c in recall) else "none"
    return RetrievalResult(product_id, query, "hybrid", mode, service_status, pool, candidates, tuple(channels))


def _run_channel(
    conn: psycopg.Connection, name: str, hits: dict[str, list[Hit]], search: Callable[[], list[Hit]]
) -> ChannelStatus:
    start = time.perf_counter()
    try:
        hits[name] = search()
    except Exception as exc:
        conn.rollback()  # a failed SQL statement must not leave the next channel an aborted transaction
        return ChannelStatus(name, "failed", f"{type(exc).__name__}: {exc}", _ms(start), 0)
    return ChannelStatus(name, "ok", None, _ms(start), len(hits[name]))


def _fuse(hits: Mapping[str, list[Hit]], params: RetrievalParams) -> tuple[Candidate, ...]:
    chunks = {h.chunk.chunk_id: h.chunk for channel_hits in hits.values() for h in channel_hits}
    ranks = {name: {h.chunk.chunk_id: (rank, h.score) for rank, h in enumerate(channel_hits, start=1)}
             for name, channel_hits in hits.items()}
    fused = rrf_fuse({name: [h.chunk.chunk_id for h in channel_hits] for name, channel_hits in hits.items()},
                     params.rrf_k)
    pool = []
    for chunk_id, rrf_score in fused[: params.rerank_pool]:
        dense = ranks.get("dense", {}).get(chunk_id, (None, None))
        bm25 = ranks.get("bm25", {}).get(chunk_id, (None, None))
        pool.append(Candidate(chunks[chunk_id], dense_rank=dense[0], dense_score=dense[1],
                              bm25_rank=bm25[0], bm25_score=bm25[1], rrf_score=rrf_score))
    return tuple(pool)


def _check_scope(result: RetrievalResult) -> None:
    stray = {c.chunk.product_id for c in result.pool + result.candidates} - {result.product_id}
    if stray:
        raise AssertionError(f"retrieval for {result.product_id} returned chunks of {sorted(stray)}")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
