from __future__ import annotations

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector

# Must match `vector(1024)` in schema.sql (BGE-M3 dense output).
EMBEDDING_DIM = 1024

# Child tables first, so this order also works for deletes.
ALL_TABLES = ("qa_runs", "annotation_labels", "annotation_items", "questions", "chunks", "reviews", "products", "datasets")


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with the pgvector type adapters registered."""
    conn = psycopg.connect(database_url)
    # The extension must exist before register_vector can look up its type.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def embedding_column_dim(conn: psycopg.Connection) -> int:
    """Dimension of the existing chunks.embedding column (CREATE TABLE IF NOT EXISTS never changes it)."""
    dim = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()[0]
    conn.rollback()
    return dim


def apply_schema(conn: psycopg.Connection) -> None:
    sql = resources.files("reviewproof").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.execute(sql)
    conn.commit()
