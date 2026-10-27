"""Local annotation review API, registered only with REVIEWPROOF_ANNOTATION=1. It shows community
answers (folded, as weak reference) but never silver labels or snippets, and withholds the agent's
draft on blind items until the human's own blind judgement is in."""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from reviewproof.annotate.pool import search_product
from reviewproof.annotate.store import Label, LabelError, get_item, list_items, save_label


class BlindBody(BaseModel):
    answerability: str
    verdict: str | None = None


class FinalBody(BaseModel):
    answerability: str
    verdict: str | None = None
    grades: dict[str, int]
    rationale: str


def annotation_router(svc: Callable) -> APIRouter:
    router = APIRouter(prefix="/v1/annotation/{label_set}")

    def load(conn, label_set: str, qid: str) -> dict:
        item = get_item(conn, label_set, qid)
        if item is None:
            raise HTTPException(404, "item not found in this label set")
        return item

    def view(conn, label_set: str, qid: str) -> dict:
        item = load(conn, label_set, qid)
        hidden = item["blind"] and "blind" not in item["labels"]
        if hidden:
            item["labels"].pop("draft", None)
        answers = conn.execute("SELECT answers FROM questions WHERE qid = %s", (qid,)).fetchone()[0]
        conn.rollback()
        return {**item, "draft_hidden": hidden,
                "community_answers": [a.get("answerText", "") for a in answers if isinstance(a, dict)]}

    def save(conn, label_set: str, label: Label) -> dict:
        try:
            save_label(conn, label_set, label)
        except LabelError as exc:
            raise HTTPException(422, str(exc)) from exc
        return view(conn, label_set, label.qid)

    @router.get("/items")
    def items(label_set: str) -> dict:
        with svc().connect() as conn:
            return {"items": list_items(conn, label_set)}

    @router.get("/items/{qid}")
    def item(label_set: str, qid: str) -> dict:
        with svc().connect() as conn:
            return view(conn, label_set, qid)

    @router.post("/items/{qid}/blind")
    def blind(label_set: str, qid: str, body: BlindBody) -> dict:
        with svc().connect() as conn:
            current = load(conn, label_set, qid)
            if not current["blind"]:
                raise HTTPException(409, "not a blind item")
            if "blind" in current["labels"]:
                raise HTTPException(409, "blind label already submitted")
            return save(conn, label_set, Label(qid, "blind", body.answerability, body.verdict, {}, "",
                                               svc().settings.annotator))

    @router.put("/items/{qid}/final")
    def final(label_set: str, qid: str, body: FinalBody) -> dict:
        with svc().connect() as conn:
            current = load(conn, label_set, qid)
            if current["blind"] and "blind" not in current["labels"]:
                raise HTTPException(409, "submit the blind label first")
            return save(conn, label_set, Label(qid, "final", body.answerability, body.verdict, body.grades,
                                               body.rationale, svc().settings.annotator))

    @router.get("/items/{qid}/search")
    def search(label_set: str, qid: str, q: str) -> dict:
        s = svc()
        with s.connect() as conn:
            current = load(conn, label_set, qid)
            with s.lock:  # BM25Cache is shared with QA runs and not thread-safe
                chunks = search_product(conn, current["product_id"], q, s.retrieval.bm25_cache,
                                        exclude={c["chunk_id"] for c in current["pool"]})
        return {"chunks": chunks}

    return router
