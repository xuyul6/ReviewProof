"""Shared test helpers. Import as `from tests.support import ...`."""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from psycopg.conninfo import conninfo_to_dict

from reviewproof.db import ALL_TABLES, EMBEDDING_DIM, apply_schema, connect
from reviewproof.ingest.amazonqa import Product, Review, review_id_for

TEST_DATABASE_URL = os.environ.get("REVIEWPROOF_TEST_DATABASE_URL")

REVIEW_TEXTS = ["Battery lasts two days.", "Too loud for me.", "Fits the 4th gen iPad."]


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def make_product(product_id: str = "A", texts: Sequence[str] = REVIEW_TEXTS) -> Product:
    reviews = tuple(
        Review(
            review_id=review_id_for(product_id, i), product_id=product_id, review_index=i,
            raw_text=t, text=t, helpful_yes=1, helpful_total=2, source_line=1, source_position=i,
        )
        for i, t in enumerate(texts)
    )
    return Product(product_id=product_id, category="Electronics", reviews=reviews)


def make_manifest(*product_ids: str) -> dict:
    return {
        "manifest_version": 1,
        "dataset_id": "test-dataset",
        "source": {"products_file": "data/p.jsonl", "products_sha256": "abc",
                   "questions_file": "data/q.jsonl", "questions_sha256": "def"},
        "rules": {},
        "products": [{"product_id": pid, "category": "Electronics", "band": "small",
                      "review_count": 3, "question_count": 1, "split": "dev"} for pid in product_ids],
        "shortfalls": [],
    }


class FakeEmbedder:
    """Deterministic unit vectors seeded by each text's SHA-256; can fail on the Nth batch."""

    def __init__(self, model_name: str = "fake", dim: int = EMBEDDING_DIM, fail_on_call: int | None = None):
        self.model_name = model_name
        self.dim = dim
        self.fail_on_call = fail_on_call
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError(f"FakeEmbedder failing on call {self.calls}")
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(self.dim)
        return (vec / np.linalg.norm(vec)).tolist()


def check_test_database_url(url: str) -> None:
    """Refuse any database not named *_test: DatabaseTestCase truncates every table."""
    dbname = conninfo_to_dict(url).get("dbname", "")
    if not dbname.endswith("_test"):
        raise RuntimeError(
            f"REVIEWPROOF_TEST_DATABASE_URL must name a database ending in _test, got {dbname!r}; "
            "database tests truncate every table")


@unittest.skipUnless(TEST_DATABASE_URL, "set REVIEWPROOF_TEST_DATABASE_URL to run database tests")
class DatabaseTestCase(unittest.TestCase):
    """Gives each test an empty schema in the test database as self.conn."""

    def setUp(self):
        check_test_database_url(TEST_DATABASE_URL)
        self.conn = connect(TEST_DATABASE_URL)
        apply_schema(self.conn)
        self.conn.execute(f"TRUNCATE {', '.join(ALL_TABLES)} CASCADE")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
