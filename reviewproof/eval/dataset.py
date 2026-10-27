"""Evaluation questions. Snippets and silver labels are read here for scoring only; nothing in this
module is ever passed to the workflow except question_text."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import psycopg


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    product_id: str
    category: str
    split: str
    strategy: str  # full_context / hybrid
    question_text: str
    question_type: str  # AmazonQA yesno / descriptive, used only to group results
    answerable_silver: bool
    snippets: tuple[str, ...]


def load_questions(conn: psycopg.Connection, split: str, full_context_max_tokens: int) -> list[EvalQuestion]:
    rows = conn.execute(
        "SELECT q.qid, q.product_id, p.category, p.split, p.token_estimate, q.question_text, q.question_type,"
        " q.is_answerable_silver, q.review_snippets FROM questions q JOIN products p USING (product_id)"
        " WHERE p.split = %s ORDER BY q.qid", (split,),
    ).fetchall()
    conn.rollback()
    return [
        EvalQuestion(qid, pid, category, sp, "full_context" if tokens <= full_context_max_tokens else "hybrid",
                     text, qtype, answerable, tuple(snippets))
        for qid, pid, category, sp, tokens, text, qtype, answerable, snippets in rows
    ]


def sample_questions(questions: Sequence[EvalQuestion], n: int, seed: int) -> list[EvalQuestion]:
    """Stratified by (category, silver label): proportional, at least one per stratum, sorted by qid."""
    if n >= len(questions):
        return sorted(questions, key=lambda q: q.qid)
    strata: dict[tuple, list[EvalQuestion]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.qid):
        strata[(q.category, q.answerable_silver)].append(q)
    keys = sorted(strata)
    total = len(questions)
    exact = {k: n * len(strata[k]) / total for k in keys}
    quota = {k: min(len(strata[k]), max(1, int(exact[k]))) for k in keys}
    while sum(quota.values()) > n:  # more strata than slots: take from the biggest quotas
        k = max((k for k in keys if quota[k] > 0), key=lambda k: (quota[k], exact[k]))
        quota[k] -= 1
    for k in sorted(keys, key=lambda k: exact[k] - int(exact[k]), reverse=True) * n:
        if sum(quota.values()) >= n:
            break
        if quota[k] < len(strata[k]):
            quota[k] += 1
    picked = []
    for category, answerable in keys:
        rng = random.Random(f"{seed}:{category}:{answerable}")
        picked += rng.sample(strata[(category, answerable)], quota[(category, answerable)])
    return sorted(picked, key=lambda q: q.qid)
