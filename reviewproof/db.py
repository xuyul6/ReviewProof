from __future__ import annotations

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector

# Must match `vector(1024)` in schema.sql (BGE-M3 dense output).
EMBEDDING_DIM = 1024

# Child tables first, so this order also works for deletes.
ALL_TABLES = ("questions", "chunks", "reviews", "products", "datasets")


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with the pgvector type adapters registered."""
    conn = psycopg.connect(database_url)
    # The extension must exist before register_vector can look up its type.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    sql = resources.files("reviewproof").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.execute(sql)
    conn.commit()
