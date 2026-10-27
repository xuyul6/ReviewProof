"""Evaluation metrics. Relevance is binary; an empty relevant set is a caller error."""
from __future__ import annotations

import math
import random
import statistics
from typing import AbstractSet, Callable, Sequence


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


# --- intervals --------------------------------------------------------------------------------

SEED = 20261027
RESAMPLES = 1000


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    return sorted_values[int(q * (len(sorted_values) - 1))]


def bootstrap_stat(n: int, stat: Callable[[Sequence[int]], float], *, seed: int = SEED,
                   resamples: int = RESAMPLES) -> tuple[float, float, float]:
    """(point, 2.5%, 97.5%) of stat over resampled row indices."""
    if n == 0:
        raise ValueError("nothing to resample")
    rng = random.Random(seed)
    point = stat(range(n))
    draws = sorted(stat([rng.randrange(n) for _ in range(n)]) for _ in range(resamples))
    return point, _percentile(draws, 0.025), _percentile(draws, 0.975)


def bootstrap_ci(values: Sequence[float], *, seed: int = SEED, resamples: int = RESAMPLES) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values is empty")
    return bootstrap_stat(len(values), lambda idx: statistics.fmean(values[i] for i in idx), seed=seed,
                          resamples=resamples)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], *, seed: int = SEED,
                   resamples: int = RESAMPLES) -> tuple[float, float, float]:
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([x - y for x, y in zip(a, b)], seed=seed, resamples=resamples)


def _interval(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


# --- answerability against the silver label ---------------------------------------------------

def _f1(rows: Sequence[dict], idx: Sequence[int]) -> float:
    tp = sum(1 for i in idx if rows[i]["answered_binary"] and rows[i]["answerable_silver"])
    fp = sum(1 for i in idx if rows[i]["answered_binary"] and not rows[i]["answerable_silver"])
    fn = sum(1 for i in idx if not rows[i]["answered_binary"] and rows[i]["answerable_silver"])
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def answerability_metrics(rows: Sequence[dict]) -> dict:
    """answered/partial count as answering; service failures are excluded and counted apart."""
    scored = [r for r in rows if r["answered_binary"] is not None]
    refused = lambda r: 0.0 if r["answered_binary"] else 1.0  # noqa: E731
    answered = [r for r in scored if r["answered_binary"]]
    answerable = [r for r in scored if r["answerable_silver"]]
    status_counts: dict[str, int] = {}
    for r in rows:
        key = "service_failed" if r["answered_binary"] is None else r["status"]
        status_counts[key] = status_counts.get(key, 0) + 1
    f1 = None
    if scored:
        point, lo, hi = bootstrap_stat(len(scored), lambda idx: _f1(scored, idx))
        f1 = {"mean": point, "lo": lo, "hi": hi, "n": len(scored)}
    return {
        "n": len(rows), "n_scored": len(scored), "n_service_failed": len(rows) - len(scored),
        "status_counts": status_counts,
        "correct_refusal_rate": _interval([refused(r) for r in scored if not r["answerable_silver"]]),
        "false_refusal_rate": _interval([refused(r) for r in answerable]),
        "answer_precision": (sum(r["answerable_silver"] for r in answered) / len(answered)) if answered else None,
        "answer_recall": (sum(r["answered_binary"] for r in answerable) / len(answerable)) if answerable else None,
        "answer_f1": f1,
        "accuracy": _interval([1.0 if r["answered_binary"] == r["answerable_silver"] else 0.0 for r in scored]),
        "service_failure_reported_as_unanswerable": sum(
            1 for r in rows if r["service_status"] == "failed" and r["status"] == "unanswerable"),
    }
