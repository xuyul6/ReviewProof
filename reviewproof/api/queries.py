"""Read-only queries shared by the HTTP API and the MCP server. None means "not found".

Community answers, snippets and answerability labels are never selected here.
"""
from __future__ import annotations

import psycopg


def list_products(conn: psycopg.Connection, *, full_context_max_tokens: int, category: str | None = None,
                  split: str | None = None) -> dict:
    rows = conn.execute(
        "SELECT product_id, category, title, review_count, chunk_count, split, token_estimate, dataset_id"
        " FROM products WHERE (%(category)s::text IS NULL OR category = %(category)s)"
        " AND (%(split)s::text IS NULL OR split = %(split)s) ORDER BY category, review_count DESC, product_id",
        {"category": category, "split": split},
    ).fetchall()
    datasets = conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id").fetchall()
    conn.rollback()
    products = [
        {"product_id": pid, "category": cat, "title": title, "review_count": reviews, "chunk_count": chunks,
         "split": sp, "strategy": "full_context" if tokens <= full_context_max_tokens else "hybrid"}
        for pid, cat, title, reviews, chunks, sp, tokens, _ in rows
    ]
    source = f"AmazonQA validation sample ({', '.join(d[0] for d in datasets)})"
    return {"source": source, "products": products}


def list_questions(conn: psycopg.Connection, product_id: str, limit: int = 20) -> dict | None:
    exists = conn.execute("SELECT 1 FROM products WHERE product_id = %s", (product_id,)).fetchone()
    rows = conn.execute(
        "SELECT qid, question_text FROM questions WHERE product_id = %s ORDER BY qid LIMIT %s", (product_id, limit)
    ).fetchall()
    conn.rollback()
    if exists is None:
        return None
    return {"product_id": product_id, "questions": [{"qid": q, "question_text": t} for q, t in rows]}


def get_review(conn: psycopg.Connection, review_id: str) -> dict | None:
    row = conn.execute(
        "SELECT review_id, product_id, text, helpful_yes, helpful_total FROM reviews WHERE review_id = %s",
        (review_id,),
    ).fetchone()
    chunks = conn.execute(
        "SELECT chunk_id, start_offset, end_offset FROM chunks WHERE review_id = %s ORDER BY chunk_id", (review_id,)
    ).fetchall()
    conn.rollback()
    if row is None:
        return None
    rid, pid, text, yes, total = row
    return {"review_id": rid, "product_id": pid, "text": text, "helpful_yes": yes, "helpful_total": total,
            "chunks": [{"chunk_id": c, "start_offset": s, "end_offset": e} for c, s, e in chunks]}
