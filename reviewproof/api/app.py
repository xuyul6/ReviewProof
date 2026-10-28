"""FastAPI app: read-only endpoints, the streaming QA endpoint, and the built Vue frontend."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from reviewproof.api import queries
from reviewproof.api.annotation import annotation_router
from reviewproof.api.qa import qa_events
from reviewproof.api.services import Services, build_services, warm_up
from reviewproof.config import load_settings
from reviewproof.runs import get_run

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class AskBody(BaseModel):
    product_id: str
    question: str


def create_app(services: Services | None = None, static_dir: Path | None = FRONTEND_DIST) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is None:  # real server: build once and load the models before the first request
            app.state.services = build_services(load_settings())
            warm_up(app.state.services)
        yield

    app = FastAPI(title="ReviewProof", lifespan=lifespan)
    if services is not None:
        app.state.services = services

    def svc() -> Services:
        return app.state.services

    if (services.settings if services is not None else load_settings()).annotation_enabled:
        app.include_router(annotation_router(svc))

    @app.get("/healthz")
    def healthz() -> dict:
        s = svc()
        try:
            conn = s.connect()
            conn.execute("SELECT 1")
            conn.close()
            database = True
        except Exception:
            database = False
        return {"database": database, "llm_model": s.llm.model_name,
                "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))}

    @app.get("/v1/products")
    def products(category: str | None = None, split: str | None = None) -> dict:
        with svc().connect() as conn:
            return queries.list_products(conn, full_context_max_tokens=svc().settings.full_context_max_tokens,
                                         category=category, split=split)

    @app.get("/v1/products/{product_id}/questions")
    def questions(product_id: str, limit: int = 20) -> dict:
        with svc().connect() as conn:
            body = queries.list_questions(conn, product_id, limit)
        if body is None:
            raise HTTPException(404, "product not found")
        return body

    @app.get("/v1/evidence/{review_id}")
    def evidence(review_id: str) -> dict:
        with svc().connect() as conn:
            body = queries.get_review(conn, review_id)
        if body is None:
            raise HTTPException(404, "review not found in the current data")
        return body

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str) -> dict:
        with svc().connect() as conn:
            body = get_run(conn, run_id)
        if body is None:
            raise HTTPException(404, "run not found")
        return body

    @app.post("/v1/qa")
    def ask(body: AskBody) -> StreamingResponse:
        if not body.question.strip():
            raise HTTPException(422, "question is empty")
        with svc().connect() as conn:
            exists = conn.execute("SELECT 1 FROM products WHERE product_id = %s", (body.product_id,)).fetchone()
        if exists is None:
            raise HTTPException(404, "product not found")

        def stream():
            for name, data in qa_events(svc(), body.product_id, body.question.strip()):
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if static_dir is not None and (static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app
