"""Retrieval evaluation: one dense and one BM25 search per question give all four variants.

The query is the raw question text (no planner rewrite), so this measures retrieval alone.
Reference evidence is AmazonQA's BM25 top-10 snippets aligned to chunks: scores measure
agreement with that BM25, not relevance, and favour BM25.
"""
from __future__ import annotations

import time
from typing import AbstractSet, Mapping, Sequence

import psycopg

from reviewproof.eval.dataset import EvalQuestion
from reviewproof.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from reviewproof.retrieval.dense import dense_search
from reviewproof.retrieval.retriever import rrf_fuse
from reviewproof.workflow.state import RetrievalDeps

VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")


def rank_variants(conn: psycopg.Connection, product_id: str, query: str, deps: RetrievalDeps
                  ) -> dict[str, list[str] | None]:
    """Ranked chunk ids per variant; a variant whose inputs failed is None (hybrid needs both channels)."""
    params = deps.params
    ranked: dict[str, list[str] | None] = {}
    try:
        hits = dense_search(conn, product_id, deps.embedder.embed_query(query), deps.embedding_model,
                            params.channel_top_k)
        ranked["dense"] = [h.chunk.chunk_id for h in hits]
    except Exception:
        conn.rollback()
        ranked["dense"] = None
    try:
        bm25_hits = deps.bm25_cache.search(conn, product_id, query, params.channel_top_k)
        ranked["bm25"] = [h.chunk.chunk_id for h in bm25_hits]
        texts = {h.chunk.chunk_id: h.chunk.text for h in bm25_hits}
    except Exception:
        conn.rollback()
        ranked["bm25"] = None
        texts = {}
    if ranked["dense"] is None or ranked["bm25"] is None:
        ranked["hybrid"] = ranked["hybrid_rerank"] = None
        return ranked
    for h in hits:
        texts[h.chunk.chunk_id] = h.chunk.text
    fused = [cid for cid, _ in rrf_fuse({"dense": ranked["dense"], "bm25": ranked["bm25"]}, params.rrf_k)]
    ranked["hybrid"] = fused
    pool = fused[: params.rerank_pool]
    try:
        if deps.reranker is None:
            raise RuntimeError("no reranker")
        scores = deps.reranker.score(query, [texts[cid] for cid in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        ranked["hybrid_rerank"] = [pool[i] for i in order] + fused[params.rerank_pool:]
    except Exception:
        ranked["hybrid_rerank"] = None
    return ranked


def _scores(ranked: Sequence[str], relevant: AbstractSet[str]) -> dict[str, float]:
    return {"recall@5": recall_at_k(ranked, relevant, 5), "recall@10": recall_at_k(ranked, relevant, 10),
            "recall@20": recall_at_k(ranked, relevant, 20), "recall@40": recall_at_k(ranked, relevant, 40),
            "hit@5": hit_at_k(ranked, relevant, 5), "mrr@10": mrr_at_k(ranked, relevant, 10),
            "ndcg@5": ndcg_at_k(ranked, relevant, 5)}


def evaluate_retrieval(conn: psycopg.Connection, questions: Sequence[EvalQuestion],
                       qrels: Mapping[str, AbstractSet[str]], deps: RetrievalDeps) -> list[dict]:
    rows = []
    for q in questions:
        relevant = qrels.get(q.qid, frozenset())
        reason = ("full_context" if q.strategy == "full_context"
                  else "silver_unanswerable" if not q.answerable_silver
                  else "no_aligned_snippet" if not relevant else None)
        row = {"qid": q.qid, "product_id": q.product_id, "category": q.category, "question_type": q.question_type,
               "strategy": q.strategy, "answerable_silver": q.answerable_silver, "n_relevant": len(relevant),
               "included": reason is None, "exclude_reason": reason}
        if reason is None:
            start = time.perf_counter()
            ranked = rank_variants(conn, q.product_id, q.question_text, deps)
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            for variant in VARIANTS:
                row[variant] = None if ranked[variant] is None else _scores(ranked[variant], relevant)
        rows.append(row)
    return rows
