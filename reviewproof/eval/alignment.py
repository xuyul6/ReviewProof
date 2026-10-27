"""Align AmazonQA review_snippets to chunks, giving silver reference evidence ("qrels").

Snippets were picked by AmazonQA with BM25, so retrieval comparisons against them favour BM25.
Matching ignores all whitespace, because AmazonQA's tokenizer put spaces around punctuation
("great !! !" for "great!!!"); about 3.4% of sampled snippets only match this way.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg

from reviewproof.eval.dataset import EvalQuestion
from reviewproof.text import normalize_text

MIN_SNIPPET_CHARS = 10  # non-space characters; shorter snippets match too easily to mean anything
_SPACE = re.compile(r"\s")


@dataclass(frozen=True)
class ReviewSpans:
    review_id: str
    text: str
    chunks: tuple[tuple[str, int, int], ...]  # (chunk_id, start_offset, end_offset) into text


def _compact(text: str) -> tuple[str, list[int]]:
    """Text without whitespace, plus each kept character's index in the original."""
    kept = [(i, ch) for i, ch in enumerate(text) if not _SPACE.match(ch)]
    return "".join(ch for _, ch in kept), [i for i, _ in kept]


def _occurrences(haystack: str, needle: str) -> Iterable[int]:
    start = haystack.find(needle)
    while start != -1:
        yield start
        start = haystack.find(needle, start + 1)


def align_snippets(snippets: Sequence[str], reviews: Sequence[ReviewSpans]) -> frozenset[str]:
    compacted = [(r, *_compact(normalize_text(r.text))) for r in reviews]
    hits: set[str] = set()
    for snippet in snippets:
        needle, _ = _compact(normalize_text(snippet))
        if len(needle) < MIN_SNIPPET_CHARS:
            continue
        for review, compact, index in compacted:
            for pos in _occurrences(compact, needle):
                start, end = index[pos], index[pos + len(needle) - 1] + 1
                inside = [c for c, s, e in review.chunks if s <= start and end <= e]
                hits.update(inside or [c for c, s, e in review.chunks if s < end and e > start])
    return frozenset(hits)


def load_review_spans(conn: psycopg.Connection, product_id: str) -> list[ReviewSpans]:
    rows = conn.execute(
        "SELECT r.review_id, r.text, c.chunk_id, c.start_offset, c.end_offset FROM reviews r"
        " JOIN chunks c USING (review_id) WHERE r.product_id = %s ORDER BY c.chunk_id", (product_id,),
    ).fetchall()
    conn.rollback()
    texts: dict[str, str] = {}
    spans: dict[str, list] = defaultdict(list)
    for review_id, text, chunk_id, start, end in rows:
        texts[review_id] = text
        spans[review_id].append((chunk_id, start, end))
    return [ReviewSpans(rid, texts[rid], tuple(spans[rid])) for rid in texts]


def silver_qrels(conn: psycopg.Connection, questions: Sequence[EvalQuestion]) -> dict[str, frozenset[str]]:
    by_product: dict[str, list[ReviewSpans]] = {}
    qrels = {}
    for q in questions:
        if q.product_id not in by_product:
            by_product[q.product_id] = load_review_spans(conn, q.product_id)
        qrels[q.qid] = align_snippets(q.snippets, by_product[q.product_id])
    return qrels
