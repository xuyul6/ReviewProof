"""BM25 over one product's chunks, with the per-product index cached in memory."""
from __future__ import annotations

import re
from collections import OrderedDict

import psycopg
from rank_bm25 import BM25Okapi

from reviewproof.retrieval.types import ChunkRow, Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def load_product_chunks(conn: psycopg.Connection, product_id: str) -> list[ChunkRow]:
    rows = conn.execute(
        "SELECT chunk_id, review_id, product_id, text FROM chunks WHERE product_id = %s ORDER BY chunk_id",
        (product_id,),
    ).fetchall()
    conn.rollback()  # end the read transaction
    return [ChunkRow(*row) for row in rows]


class _ProductIndex:
    def __init__(self, chunks: list[ChunkRow]):
        self.chunks = chunks
        self.token_sets = [set(tokens) for tokens in (tokenize(c.text) for c in chunks)]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])


class BM25Cache:
    """Keyed by (product_id, chunk count, total text length), so a re-import rebuilds the index."""

    def __init__(self, max_products: int = 64):
        self.max_products = max_products
        self._indexes: OrderedDict[tuple, _ProductIndex] = OrderedDict()

    def search(self, conn: psycopg.Connection, product_id: str, query: str, top_k: int) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        index = self._index(conn, product_id)
        if index is None:
            return []
        scores = index.bm25.get_scores(terms)
        query_terms = set(terms)
        # A hit must share a term with the query; scores alone can't tell, because rank_bm25
        # gives terms that appear in most chunks a negative IDF when a product has few chunks.
        matched = [i for i, tokens in enumerate(index.token_sets) if tokens & query_terms]
        matched.sort(key=lambda i: (-scores[i], index.chunks[i].chunk_id))
        return [Hit(index.chunks[i], float(scores[i])) for i in matched[:top_k]]

    def _index(self, conn: psycopg.Connection, product_id: str) -> _ProductIndex | None:
        count, length = conn.execute(
            "SELECT count(*), coalesce(sum(length(text)), 0) FROM chunks WHERE product_id = %s", (product_id,)
        ).fetchone()
        conn.rollback()
        if count == 0:
            return None  # BM25Okapi divides by zero on an empty corpus
        key = (product_id, count, length)
        if key in self._indexes:
            self._indexes.move_to_end(key)
            return self._indexes[key]
        for stale in [k for k in self._indexes if k[0] == product_id]:
            del self._indexes[stale]
        index = _ProductIndex(load_product_chunks(conn, product_id))
        self._indexes[key] = index
        while len(self._indexes) > self.max_products:
            self._indexes.popitem(last=False)
        return index
