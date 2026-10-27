"""Write a sampled dataset into Postgres. Safe to re-run: every table is upserted.

Questions, community answers and answerability labels go only into `questions`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import psycopg
from psycopg.types.json import Jsonb

from reviewproof.ingest.amazonqa import Product, Question
from reviewproof.ingest.chunking import chunk_review


@dataclass(frozen=True)
class LoadReport:
    products: int
    reviews: int
    chunks: int
    questions: int


_UPSERT_DATASET = """
INSERT INTO datasets (dataset_id, source_file, source_sha256, manifest)
VALUES (%s, %s, %s, %s)
ON CONFLICT (dataset_id) DO UPDATE SET source_file = EXCLUDED.source_file,
  source_sha256 = EXCLUDED.source_sha256, manifest = EXCLUDED.manifest, loaded_at = now()
"""

_UPSERT_PRODUCT = """
INSERT INTO products (product_id, dataset_id, category, split, band, review_count, chunk_count, token_estimate)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (product_id) DO UPDATE SET dataset_id = EXCLUDED.dataset_id, category = EXCLUDED.category,
  split = EXCLUDED.split, band = EXCLUDED.band, review_count = EXCLUDED.review_count,
  chunk_count = EXCLUDED.chunk_count, token_estimate = EXCLUDED.token_estimate
"""

_UPSERT_REVIEW = """
INSERT INTO reviews (review_id, product_id, review_index, raw_text, text, helpful_yes, helpful_total,
  source_line, source_position)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (review_id) DO UPDATE SET product_id = EXCLUDED.product_id, review_index = EXCLUDED.review_index,
  raw_text = EXCLUDED.raw_text, text = EXCLUDED.text, helpful_yes = EXCLUDED.helpful_yes,
  helpful_total = EXCLUDED.helpful_total, source_line = EXCLUDED.source_line,
  source_position = EXCLUDED.source_position
"""

# A chunk keeps its embedding only while its text is unchanged.
_UPSERT_CHUNK = """
INSERT INTO chunks (chunk_id, review_id, product_id, chunk_index, text, start_offset, end_offset, token_estimate)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET review_id = EXCLUDED.review_id, product_id = EXCLUDED.product_id,
  chunk_index = EXCLUDED.chunk_index, text = EXCLUDED.text, start_offset = EXCLUDED.start_offset,
  end_offset = EXCLUDED.end_offset, token_estimate = EXCLUDED.token_estimate,
  embedding = CASE WHEN chunks.text = EXCLUDED.text THEN chunks.embedding END,
  embedding_model = CASE WHEN chunks.text = EXCLUDED.text THEN chunks.embedding_model END
"""

_UPSERT_QUESTION = """
INSERT INTO questions (qid, product_id, question_text, question_type, is_answerable_silver, answers, review_snippets)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (qid) DO UPDATE SET product_id = EXCLUDED.product_id, question_text = EXCLUDED.question_text,
  question_type = EXCLUDED.question_type, is_answerable_silver = EXCLUDED.is_answerable_silver,
  answers = EXCLUDED.answers, review_snippets = EXCLUDED.review_snippets
"""


def load_dataset(
    conn: psycopg.Connection,
    manifest: Mapping,
    products: Mapping[str, Product],
    questions: Mapping[str, Sequence[Question]],
) -> LoadReport:
    entries = manifest["products"]
    missing = [e["product_id"] for e in entries if e["product_id"] not in products]
    if missing:
        raise ValueError(f"manifest products missing from the source file: {', '.join(missing)}")

    dataset_id = manifest["dataset_id"]
    source = manifest["source"]
    try:
        with conn.cursor() as cur:
            cur.execute(_UPSERT_DATASET, (dataset_id, source["products_file"], source["products_sha256"], Jsonb(dict(manifest))))
            for entry in entries:
                product = products[entry["product_id"]]
                chunks = [c for review in product.reviews for c in chunk_review(review)]
                cur.execute(_UPSERT_PRODUCT, (
                    product.product_id, dataset_id, product.category, entry["split"], entry["band"],
                    len(product.reviews), len(chunks), sum(c.token_estimate for c in chunks),
                ))
                cur.executemany(_UPSERT_REVIEW, [
                    (r.review_id, r.product_id, r.review_index, r.raw_text, r.text, r.helpful_yes,
                     r.helpful_total, r.source_line, r.source_position)
                    for r in product.reviews
                ])
                cur.executemany(_UPSERT_CHUNK, [
                    (c.chunk_id, c.review_id, c.product_id, c.chunk_index, c.text, c.start_offset,
                     c.end_offset, c.token_estimate)
                    for c in chunks
                ])
                product_questions = list(questions.get(product.product_id, ()))
                cur.executemany(_UPSERT_QUESTION, [
                    (q.qid, q.product_id, q.question_text, q.question_type, q.is_answerable,
                     Jsonb(list(q.answers)), Jsonb(list(q.review_snippets)))
                    for q in product_questions
                ])
                # Drop rows from an earlier load that this load no longer produces,
                # so the stored counts always match the products row.
                for table, key, ids in (
                    ("reviews", "review_id", [r.review_id for r in product.reviews]),
                    ("chunks", "chunk_id", [c.chunk_id for c in chunks]),
                    ("questions", "qid", [q.qid for q in product_questions]),
                ):
                    cur.execute(
                        f"DELETE FROM {table} WHERE product_id = %s AND NOT ({key} = ANY(%s))",
                        (product.product_id, ids),
                    )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    ids = [e["product_id"] for e in entries]
    counts = [
        conn.execute(f"SELECT count(*) FROM {table} WHERE product_id = ANY(%s)", (ids,)).fetchone()[0]
        for table in ("products", "reviews", "chunks", "questions")
    ]
    return LoadReport(*counts)
