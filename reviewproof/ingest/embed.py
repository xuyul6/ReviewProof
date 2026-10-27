"""Embed chunks that have no vector yet (or one from a different model).

Each batch is committed on its own, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import math
from typing import Callable, Collection

import numpy as np
import psycopg

from reviewproof.db import EMBEDDING_DIM
from reviewproof.embedding import Embedder

_PENDING = "embedding IS NULL OR embedding_model IS DISTINCT FROM %s"


def embed_pending_chunks(
    conn: psycopg.Connection,
    embedder: Embedder,
    batch_size: int,
    product_ids: Collection[str] | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Embed pending chunks batch by batch; on_batch(done, total_pending) runs after each commit."""
    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(f"embedder dim {embedder.dim} does not match the schema's vector({EMBEDDING_DIM})")

    where = f"({_PENDING})"
    params: list = [embedder.model_name]
    if product_ids is not None:
        where += " AND product_id = ANY(%s)"
        params.append(list(product_ids))
    total = conn.execute(f"SELECT count(*) FROM chunks WHERE {where}", params).fetchone()[0]
    query = f"SELECT chunk_id, text FROM chunks WHERE {where} ORDER BY chunk_id LIMIT %s"
    params.append(batch_size)

    written = 0
    while True:
        rows = conn.execute(query, params).fetchall()
        conn.rollback()  # end the read transaction; nothing written yet
        if not rows:
            return written
        vectors = embedder.embed_documents([text for _, text in rows])
        if len(vectors) != len(rows):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(rows)} texts")
        for vec in vectors:
            if len(vec) != EMBEDDING_DIM or not all(math.isfinite(x) for x in vec):
                raise ValueError("embedder returned a vector with the wrong length or non-finite values")
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s, embedding_model = %s WHERE chunk_id = %s",
                [(np.asarray(vec, dtype=np.float32), embedder.model_name, chunk_id)
                 for (chunk_id, _), vec in zip(rows, vectors)],
            )
        conn.commit()
        written += len(rows)
        if on_batch is not None:
            on_batch(written, total)
