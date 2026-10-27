"""Read AmazonQA product records (val-qar_products.jsonl) and questions (val-qar.jsonl).

Review IDs are generated here (`{asin}-r{index:05d}`); AmazonQA has no native review IDs,
so these must never be presented as Amazon review IDs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

from reviewproof.text import dedupe_key, normalize_text


@dataclass(frozen=True)
class Review:
    review_id: str
    product_id: str
    review_index: int
    raw_text: str
    text: str
    helpful_yes: int
    helpful_total: int
    source_line: int
    source_position: int


@dataclass(frozen=True)
class Product:
    product_id: str
    category: str
    reviews: tuple[Review, ...]


@dataclass(frozen=True)
class Question:
    qid: str
    product_id: str
    question_text: str
    question_type: str
    is_answerable: bool
    answers: tuple[dict, ...]
    review_snippets: tuple[str, ...]


def review_id_for(product_id: str, review_index: int) -> str:
    return f"{product_id}-r{review_index:05d}"


def _helpful(raw: object) -> tuple[int, int]:
    if isinstance(raw, list) and len(raw) == 2 and all(isinstance(v, int) for v in raw):
        return raw[0], raw[1]
    return 0, 0


def load_products(path: Path, product_ids: Collection[str]) -> dict[str, Product]:
    """Load the requested products, merging records that repeat an ASIN.

    The first record's reviews come first; later records only add reviews whose
    dedupe_key is new. Duplicates inside a single record are kept (different
    buyers can write the same short review). Empty reviews are skipped and do not
    consume an index.
    """
    wanted = set(product_ids)
    categories: dict[str, str] = {}
    # Per product: (raw_text, text, helpful, source_line, source_position) in final order.
    kept: dict[str, list[tuple[str, str, tuple[int, int], int, int]]] = {}
    seen_keys: dict[str, set[str]] = {}

    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            record = json.loads(line)
            asin = record.get("asin")
            if asin not in wanted:
                continue
            first_record = asin not in kept
            if first_record:
                categories[asin] = record.get("category", "")
                kept[asin] = []
                seen_keys[asin] = set()
            record_keys: set[str] = set()
            for position, review in enumerate(record.get("reviews") or []):
                raw_text = review.get("reviewText") or ""
                text = normalize_text(raw_text)
                if not text:
                    continue
                key = dedupe_key(text)
                if not first_record and key in seen_keys[asin]:
                    continue
                record_keys.add(key)
                kept[asin].append((raw_text, text, _helpful(review.get("helpful")), line_no, position))
            seen_keys[asin] |= record_keys

    products: dict[str, Product] = {}
    for asin, rows in kept.items():
        reviews = tuple(
            Review(
                review_id=review_id_for(asin, index),
                product_id=asin,
                review_index=index,
                raw_text=raw_text,
                text=text,
                helpful_yes=helpful[0],
                helpful_total=helpful[1],
                source_line=line_no,
                source_position=position,
            )
            for index, (raw_text, text, helpful, line_no, position) in enumerate(rows)
        )
        products[asin] = Product(product_id=asin, category=categories[asin], reviews=reviews)
    return products


def load_questions(path: Path, product_ids: Collection[str]) -> dict[str, list[Question]]:
    wanted = set(product_ids)
    questions: dict[str, list[Question]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            asin = row.get("asin")
            if asin not in wanted:
                continue
            questions.setdefault(asin, []).append(
                Question(
                    qid=str(row["qid"]),
                    product_id=asin,
                    question_text=row.get("questionText", ""),
                    question_type=row.get("questionType", ""),
                    is_answerable=bool(row.get("is_answerable")),
                    answers=tuple(row.get("answers") or ()),
                    review_snippets=tuple(row.get("review_snippets") or ()),
                )
            )
    return questions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
