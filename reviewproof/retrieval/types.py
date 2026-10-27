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
