"""Gold labels exported by `python -m reviewproof.annotate export` (annotations/<set>/labels.jsonl):
loading with checks against the current database, qrels, and agreement statistics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import psycopg

from reviewproof.eval.metrics import cohen_kappa


class GoldError(ValueError):
    pass


@dataclass(frozen=True)
class GoldLabel:
    qid: str
    split: str
    product_id: str
    question_type: str
    answerability: str  # answered / partial / unanswerable
    verdict: str | None
    grades: Mapping[str, int]  # chunk_id -> 0/1/2
    draft: dict | None  # agent draft snapshot {answerability, verdict, grades}
    blind: dict | None  # human blind judgement {answerability, verdict}
    changed_from_draft: bool = True

    @property
    def answerable(self) -> bool:
        return self.answerability != "unanswerable"


def load_gold(conn: psycopg.Connection, path: Path) -> dict[str, GoldLabel]:
    """Refuse labels made on other data: a dataset_id not in the database, a question that is gone,
    or a graded chunk that no longer exists in that product (it would silently count as irrelevant)."""
    if not path.exists():
        raise GoldError(f"gold label file not found: {path}")
    datasets = {d for (d,) in conn.execute("SELECT dataset_id FROM datasets").fetchall()}
    labels: dict[str, GoldLabel] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            label = GoldLabel(qid=r["qid"], split=r["split"], product_id=r["product_id"],
                              question_type=r["question_type"], answerability=r["answerability"],
                              verdict=r["verdict"], grades=r["grades"], draft=r.get("draft"), blind=r.get("blind"),
                              changed_from_draft=r.get("changed_from_draft", True))
            dataset_id = r["dataset_id"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise GoldError(f"{path}:{line_no}: malformed label ({exc})") from exc
        if dataset_id not in datasets:
            raise GoldError(f"{label.qid}: labelled on dataset {dataset_id!r}, database has {sorted(datasets)}")
        found = conn.execute("SELECT 1 FROM questions WHERE qid = %s AND product_id = %s",
                             (label.qid, label.product_id)).fetchone()
        if found is None:
            raise GoldError(f"{label.qid}: question not found for product {label.product_id}")
        present = {cid for (cid,) in conn.execute(
            "SELECT chunk_id FROM chunks WHERE product_id = %s AND chunk_id = ANY(%s)",
            (label.product_id, list(label.grades))).fetchall()}
        missing = sorted(set(label.grades) - present)
        if missing:
            raise GoldError(f"{label.qid}: graded chunks not in the current data: {missing[:5]}")
        labels[label.qid] = label
    conn.rollback()
    return labels


def gold_qrels(labels: Mapping[str, GoldLabel]) -> dict[str, dict[str, int]]:
    """Grades for answered and partial questions; unanswerable questions have no relevant evidence."""
    return {qid: dict(lab.grades) for qid, lab in labels.items() if lab.answerable}


def _pairwise(a: list[str], b: list[str]) -> dict:
    return {"n": len(a), "kappa": cohen_kappa(a, b),
            "agreement": (sum(x == y for x, y in zip(a, b)) / len(a)) if a else None}


def agreement(labels: Mapping[str, GoldLabel], silver: Mapping[str, bool]) -> dict:
    """Gold vs AmazonQA silver (binary), agent draft vs final review and human blind vs agent draft
    (three states). The blind comparison is human vs agent, not two humans."""
    qids = sorted(q for q in labels if q in silver)
    gold_bin = ["answerable" if labels[q].answerable else "unanswerable" for q in qids]
    silver_bin = ["answerable" if silver[q] else "unanswerable" for q in qids]
    drafted = [lab for lab in labels.values() if lab.draft]
    blind = [lab for lab in labels.values() if lab.draft and lab.blind]
    return {
        "gold_vs_silver": _pairwise(gold_bin, silver_bin),
        "draft_vs_final": {**_pairwise([lab.draft["answerability"] for lab in drafted],
                                       [lab.answerability for lab in drafted]),
                           "changed_share": (sum(lab.changed_from_draft for lab in drafted) / len(drafted))
                           if drafted else None},
        "blind_vs_draft": _pairwise([lab.blind["answerability"] for lab in blind],
                                    [lab.draft["answerability"] for lab in blind]),
    }
