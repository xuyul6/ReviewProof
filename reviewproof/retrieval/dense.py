"""Exact per-product vector search: pgvector sorts one product's chunks, no ANN index."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import psycopg

from reviewproof.retrieval.types import ChunkRow, Hit


class IncompleteEmbeddingsError(RuntimeError):
    """Some of the product's chunks have no vector from the requested model."""

    def __init__(self, product_id: str, missing: int, embedding_model: str):
        super().__init__(f"{missing} chunks of {product_id} have no {embedding_model} vector")
        self.missing = missing


def dense_search(
    conn: psycopg.Connection,
    product_id: str,
    query_vector: Sequence[float],
    embedding_model: str,
    top_k: int,
) -> list[Hit]:
    try:
        missing = conn.execute(
            "SELECT count(*) FROM chunks WHERE product_id = %s"
            " AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %s)",
            (product_id, embedding_model),
        ).fetchone()[0]
        if missing:
            raise IncompleteEmbeddingsError(product_id, missing, embedding_model)
        rows = conn.execute(
            "SELECT chunk_id, review_id, product_id, text, 1 - (embedding <=> %s) AS score"
            " FROM chunks WHERE product_id = %s ORDER BY embedding <=> %s, chunk_id LIMIT %s",
            (vec := np.asarray(query_vector, dtype=np.float32), product_id, vec, top_k),
        ).fetchall()
    finally:
        conn.rollback()  # end the read transaction, aborted or not
    return [Hit(ChunkRow(*row[:4]), float(row[4])) for row in rows]
