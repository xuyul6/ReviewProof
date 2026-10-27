"""Evaluation metrics. Relevance is binary; an empty relevant set is a caller error."""
from __future__ import annotations

import math
from typing import AbstractSet, Sequence


def _check(relevant: AbstractSet[str]) -> None:
    if not relevant:
        raise ValueError("relevant set is empty; such questions are excluded before scoring")


def recall_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    _check(relevant)
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    _check(relevant)
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    _check(relevant)
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    _check(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg
