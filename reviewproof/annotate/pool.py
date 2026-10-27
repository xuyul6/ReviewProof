"""Which questions get gold labels, which chunks annotators judge for each (the candidate pool),
and the in-product search annotators use to look beyond the pool."""
from __future__ import annotations

import random
from typing import AbstractSet, Mapping, Sequence

import psycopg

from reviewproof.eval.dataset import EvalQuestion, sample_questions
from reviewproof.eval.retrieval_eval import VARIANTS, rank_variants
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.workflow.state import RetrievalDeps

POOL_DEPTH = 10
STRATA = ("category", "question_type", "answerable_silver")


def build_pool(conn: psycopg.Connection, q: EvalQuestion, deps: RetrievalDeps) -> tuple[str, ...]:
    """Every chunk for full-context products; otherwise the union of each variant's top POOL_DEPTH.
    Sorted by chunk_id so annotators cannot tell which system ranked a chunk first."""
    if q.strategy == "full_context":
        rows = conn.execute("SELECT chunk_id FROM chunks WHERE product_id = %s", (q.product_id,)).fetchall()
        conn.rollback()
        return tuple(sorted(r[0] for r in rows))
    ranked = rank_variants(conn, q.product_id, q.question_text, deps)
    usable = [ranked[v] for v in VARIANTS if ranked.get(v) is not None]
    if not usable:
        raise RuntimeError(f"{q.qid}: every retrieval variant failed, no pool")
    return tuple(sorted({cid for ids in usable for cid in ids[:POOL_DEPTH]}))


def pick_items(questions_by_split: Mapping[str, Sequence[EvalQuestion]], counts: Mapping[str, int], *,
               blind_fraction: float, seed: int) -> list[tuple[EvalQuestion, bool]]:
    picked = []
    for split, n in counts.items():
        sample = sample_questions(questions_by_split[split], n, seed, by=STRATA)
        blind = set(random.Random(f"{seed}:blind:{split}").sample([q.qid for q in sample],
                                                                  round(len(sample) * blind_fraction)))
        picked += [(q, q.qid in blind) for q in sample]
    return picked


def search_product(conn: psycopg.Connection, product_id: str, query: str, bm25_cache: BM25Cache, *,
                   exclude: AbstractSet[str], limit: int = 20) -> list[dict]:
    """BM25 top `limit` plus case-insensitive substring matches, minus `exclude`, BM25 hits first."""
    found: dict[str, dict] = {}
    for hit in bm25_cache.search(conn, product_id, query, limit + len(exclude)):
        c = hit.chunk
        if c.chunk_id not in exclude and len(found) < limit:
            found[c.chunk_id] = {"chunk_id": c.chunk_id, "review_id": c.review_id, "text": c.text}
    if query.strip():
        rows = conn.execute(
            "SELECT chunk_id, review_id, text FROM chunks WHERE product_id = %s AND strpos(lower(text), lower(%s)) > 0"
            " AND NOT (chunk_id = ANY(%s)) ORDER BY chunk_id LIMIT %s",
            (product_id, query.strip(), list(exclude), limit)).fetchall()
        conn.rollback()
        for cid, rid, text in rows:
            found.setdefault(cid, {"chunk_id": cid, "review_id": rid, "text": text})
    return list(found.values())
