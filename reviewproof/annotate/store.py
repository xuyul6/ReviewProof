"""Gold annotation storage: items (question + fixed candidate pool) and labels from three sources
(agent draft, human blind judgement, human final review). Labels are checked against
docs/annotation_guidelines_cn.md before anything is written. Nothing here reaches a prompt."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import AbstractSet, Mapping, Sequence

import psycopg
from psycopg.types.json import Jsonb

GUIDELINE_VERSION = "gold-v1"
ANSWERABILITY = ("answered", "partial", "unanswerable")
VERDICTS = ("yes", "no", "mixed", "unknown")
GRADES = (0, 1, 2)
SOURCES = ("draft", "blind", "final")


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class ItemSpec:
    qid: str
    split: str
    blind: bool
    pool: tuple[str, ...]


@dataclass(frozen=True)
class Label:
    qid: str
    source: str  # draft / blind / final
    answerability: str
    verdict: str | None  # yes/no questions only
    grades: Mapping[str, int]  # chunk_id -> 0/1/2; empty for blind labels
    rationale: str
    annotator: str


def validate_label(label: Label, *, pool: Sequence[str], product_chunk_ids: AbstractSet[str],
                   question_type: str) -> None:
    if label.source not in SOURCES:
        raise LabelError(f"unknown source {label.source!r}")
    if not isinstance(label.rationale, str):
        raise LabelError("rationale must be text")
    if not isinstance(label.grades, Mapping) or any(
            not isinstance(cid, str) or type(g) is not int for cid, g in label.grades.items()):
        raise LabelError("grades must be an object mapping chunk ids to 0, 1 or 2")
    if label.answerability not in ANSWERABILITY:
        raise LabelError(f"answerability must be one of {ANSWERABILITY}, got {label.answerability!r}")
    if question_type == "yesno":
        if label.verdict not in VERDICTS:
            raise LabelError(f"yes/no question needs a verdict in {VERDICTS}, got {label.verdict!r}")
        if (label.verdict == "unknown") != (label.answerability == "unanswerable"):
            raise LabelError("verdict 'unknown' goes with answerability 'unanswerable', and only with it")
    elif label.verdict is not None:
        raise LabelError(f"only yes/no questions take a verdict ({question_type} question)")
    if label.source == "blind":
        return
    if not label.rationale.strip():
        raise LabelError("rationale is empty")
    foreign = sorted(set(label.grades) - set(product_chunk_ids))
    if foreign:
        raise LabelError(f"chunks not in this product: {foreign}")
    missing = sorted(set(pool) - set(label.grades))
    if missing:
        raise LabelError(f"pool chunks without a grade: {missing}")
    bad = sorted(cid for cid, g in label.grades.items() if g not in GRADES)
    if bad:
        raise LabelError(f"grades must be 0, 1 or 2: {bad}")
    has_two = any(g == 2 for g in label.grades.values())
    if label.answerability == "unanswerable" and has_two:
        raise LabelError("unanswerable questions cannot have grade-2 evidence")
    if label.answerability != "unanswerable" and not has_two:
        raise LabelError(f"{label.answerability} needs at least one grade-2 chunk")


def add_items(conn: psycopg.Connection, label_set: str, items: Sequence[ItemSpec]) -> None:
    try:
        exists = conn.execute("SELECT 1 FROM annotation_items WHERE label_set = %s LIMIT 1", (label_set,)).fetchone()
        if exists:
            raise LabelError(f"label set {label_set!r} already exists")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO annotation_items (label_set, qid, dataset_id, split, blind, pool)"
                " SELECT %s, q.qid, p.dataset_id, %s, %s, %s FROM questions q JOIN products p USING (product_id)"
                " WHERE q.qid = %s",
                [(label_set, i.split, i.blind, Jsonb(list(i.pool)), i.qid) for i in items])
        stored = conn.execute("SELECT count(*) FROM annotation_items WHERE label_set = %s", (label_set,)).fetchone()[0]
        if stored != len(items):
            raise LabelError(f"{len(items) - stored} qids are not in the questions table")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _status(sources: AbstractSet[str]) -> str:
    return "done" if "final" in sources else "drafted" if "draft" in sources else "pending"


def list_items(conn: psycopg.Connection, label_set: str) -> list[dict]:
    rows = conn.execute(
        "SELECT i.qid, i.split, i.blind, coalesce(array_agg(l.source) FILTER (WHERE l.source IS NOT NULL), '{}')"
        " FROM annotation_items i LEFT JOIN annotation_labels l USING (label_set, qid)"
        " WHERE i.label_set = %s GROUP BY i.qid, i.split, i.blind ORDER BY i.qid", (label_set,)).fetchall()
    conn.rollback()
    return [{"qid": qid, "split": split, "blind": blind, "status": _status(set(sources))}
            for qid, split, blind, sources in rows]


def _labels(conn: psycopg.Connection, label_set: str, qid: str) -> dict[str, Label]:
    rows = conn.execute(
        "SELECT source, answerability, verdict, grades, rationale, annotator FROM annotation_labels"
        " WHERE label_set = %s AND qid = %s", (label_set, qid)).fetchall()
    return {src: Label(qid, src, ans, verdict, grades, rationale, annotator)
            for src, ans, verdict, grades, rationale, annotator in rows}


def get_item(conn: psycopg.Connection, label_set: str, qid: str) -> dict | None:
    row = conn.execute(
        "SELECT q.qid, q.product_id, p.category, i.split, q.question_text, q.question_type, i.blind, i.pool"
        " FROM annotation_items i JOIN questions q USING (qid) JOIN products p USING (product_id)"
        " WHERE i.label_set = %s AND i.qid = %s", (label_set, qid)).fetchone()
    if row is None:
        conn.rollback()
        return None
    qid, product_id, category, split, text, qtype, blind, pool = row
    chunks = {cid: (rid, t) for cid, rid, t in conn.execute(
        "SELECT chunk_id, review_id, text FROM chunks WHERE chunk_id = ANY(%s)", (pool,)).fetchall()}
    labels = _labels(conn, label_set, qid)
    conn.rollback()
    return {"qid": qid, "product_id": product_id, "category": category, "split": split, "question_text": text,
            "question_type": qtype, "blind": blind,
            "pool": [{"chunk_id": cid, "review_id": chunks[cid][0], "text": chunks[cid][1]}
                     for cid in pool if cid in chunks],
            # Pool chunks gone from the database (data re-ingested since pick); the item cannot be saved.
            "missing_chunks": [cid for cid in pool if cid not in chunks],
            "labels": {src: asdict(lab) for src, lab in labels.items()}}


def save_label(conn: psycopg.Connection, label_set: str, label: Label) -> None:
    try:
        row = conn.execute(
            "SELECT i.blind, i.pool, q.question_type, q.product_id FROM annotation_items i JOIN questions q USING (qid)"
            " WHERE i.label_set = %s AND i.qid = %s", (label_set, label.qid)).fetchone()
        if row is None:
            raise LabelError(f"{label.qid} is not in label set {label_set!r}")
        blind, pool, qtype, product_id = row
        product_chunks = {cid for (cid,) in conn.execute(
            "SELECT chunk_id FROM chunks WHERE product_id = %s", (product_id,)).fetchall()}
        validate_label(label, pool=pool, product_chunk_ids=product_chunks, question_type=qtype)
        existing = _labels(conn, label_set, label.qid)
        if label.source == "blind":
            if not blind:
                raise LabelError(f"{label.qid} is not a blind item")
            if "blind" in existing:
                raise LabelError(f"{label.qid} already has its blind label")
        if label.source == "draft" and ("blind" in existing or "final" in existing):
            raise LabelError(f"{label.qid}: human labels exist, the draft they were compared with cannot change")
        if label.source == "final" and blind and "blind" not in existing:
            raise LabelError(f"{label.qid} is a blind item: submit the blind label first")
        conn.execute(
            "INSERT INTO annotation_labels (label_set, qid, source, answerability, verdict, grades, rationale,"
            " annotator, guideline_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (label_set, qid, source) DO UPDATE SET answerability = EXCLUDED.answerability,"
            " verdict = EXCLUDED.verdict, grades = EXCLUDED.grades, rationale = EXCLUDED.rationale,"
            " annotator = EXCLUDED.annotator, guideline_version = EXCLUDED.guideline_version, created_at = now()",
            (label_set, label.qid, label.source, label.answerability, label.verdict, Jsonb(dict(label.grades)),
             label.rationale, label.annotator, GUIDELINE_VERSION))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def export_labels(conn: psycopg.Connection, label_set: str) -> list[dict]:
    items = conn.execute(
        "SELECT i.qid, i.dataset_id, i.split, i.pool, q.product_id, q.question_type FROM annotation_items i"
        " JOIN questions q USING (qid) WHERE i.label_set = %s ORDER BY i.qid", (label_set,)).fetchall()
    out = []
    for qid, dataset_id, split, pool, product_id, qtype in items:
        labels = _labels(conn, label_set, qid)
        final = labels.get("final")
        if final is None:
            continue
        draft, blind = labels.get("draft"), labels.get("blind")
        draft_snap = None if draft is None else {"answerability": draft.answerability, "verdict": draft.verdict,
                                                 "grades": dict(draft.grades)}
        out.append({
            "label_set": label_set, "dataset_id": dataset_id, "guideline_version": GUIDELINE_VERSION,
            "qid": qid, "product_id": product_id, "split": split, "question_type": qtype,
            "answerability": final.answerability, "verdict": final.verdict, "grades": dict(final.grades),
            "rationale": final.rationale, "annotator": final.annotator, "pool": list(pool),
            "draft": draft_snap,
            "blind": None if blind is None else {"answerability": blind.answerability, "verdict": blind.verdict},
            "changed_from_draft": draft_snap is None or (draft.answerability, draft.verdict, dict(draft.grades))
            != (final.answerability, final.verdict, dict(final.grades)),
        })
    conn.rollback()
    return out
