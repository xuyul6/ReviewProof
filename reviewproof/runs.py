"""Persist and look up QA runs (`qa_runs`), so any answer can be traced back by run_id."""
from __future__ import annotations

import uuid
from dataclasses import asdict

import psycopg
from psycopg.types.json import Jsonb

from reviewproof.workflow.graph import QAResult


def new_run_id() -> str:
    return uuid.uuid4().hex


def save_run(conn: psycopg.Connection, run_id: str, result: QAResult, llm_model: str) -> None:
    try:
        conn.execute(
            "INSERT INTO qa_runs (run_id, product_id, question, route, status, service_status, verdict,"
            " llm_model, latency_ms, result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (run_id, result.product_id, result.question, result.route, result.status, result.service_status,
             result.verdict, llm_model, result.latency_ms, Jsonb(asdict(result))),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_run(conn: psycopg.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT created_at, llm_model, result FROM qa_runs WHERE run_id = %s", (run_id,)).fetchone()
    conn.rollback()
    if row is None:
        return None
    created_at, llm_model, result = row
    return {"run_id": run_id, "created_at": created_at.isoformat(), "llm_model": llm_model, **result}
