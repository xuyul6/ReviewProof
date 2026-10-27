"""Build the fixed sampling manifest (samples/manifest_v1.json).

Products are stratified by review count so that retrieval has real work to do on
large products, while small products exercise the full-context strategy.

Usage:
    python -m reviewproof.ingest.sampling --products data/val-qar_products.jsonl \
        --questions data/val-qar.jsonl --output samples/manifest_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from reviewproof.ingest.amazonqa import file_sha256
from reviewproof.text import normalize_text

BANDS = ("large", "medium", "small")
BAND_RULES = {"large": ">=200", "medium": "50-199", "small": "1-49"}
DATASET_ID = "amazonqa-val-v1"

DEFAULT_CATEGORIES = ("Electronics", "Home_and_Kitchen", "Automotive", "Beauty")
DEFAULT_PER_BAND = {"large": 10, "medium": 10, "small": 5}
DEFAULT_MIN_QUESTIONS = 5
DEFAULT_SEED = 20261027
DEFAULT_DEV_FRACTION = 0.3


@dataclass(frozen=True)
class ProductStats:
    product_id: str
    category: str
    review_count: int
    question_count: int


def band_for(review_count: int) -> str:
    if review_count < 1:
        raise ValueError(f"review_count must be >= 1, got {review_count}")
    if review_count >= 200:
        return "large"
    if review_count >= 50:
        return "medium"
    return "small"


def scan_product_stats(products_path: Path, questions_path: Path) -> list[ProductStats]:
    """Stats for products present in both files.

    review_count is the largest non-empty review count over an ASIN's records; it is
    only used for banding (the loaded count after merging is authoritative).
    """
    review_counts: dict[str, int] = {}
    categories: dict[str, str] = {}
    with open(products_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            asin = record["asin"]
            count = sum(1 for r in record.get("reviews") or [] if normalize_text(r.get("reviewText") or ""))
            categories.setdefault(asin, record.get("category", ""))
            review_counts[asin] = max(count, review_counts.get(asin, 0))

    question_counts: Counter[str] = Counter()
    with open(questions_path, encoding="utf-8") as fh:
        for line in fh:
            question_counts[json.loads(line)["asin"]] += 1

    return [
        ProductStats(asin, categories[asin], review_counts[asin], question_counts[asin])
        for asin in sorted(review_counts)
        if question_counts[asin] > 0
    ]


def select_sample(
    stats: Sequence[ProductStats],
    categories: Sequence[str],
    per_band: Mapping[str, int],
    min_questions: int,
    seed: int,
) -> tuple[list[ProductStats], list[dict]]:
    """Pick per_band products per category; a band's shortfall carries to the next band."""
    selected: list[ProductStats] = []
    shortfalls: list[dict] = []
    for category in categories:
        rng = random.Random(f"{seed}:{category}")
        eligible = [
            s for s in stats
            if s.category == category and s.question_count >= min_questions and s.review_count >= 1
        ]
        carry = 0
        for band in BANDS:
            requested = per_band[band] + carry
            pool = sorted((s for s in eligible if band_for(s.review_count) == band), key=lambda s: s.product_id)
            picked = rng.sample(pool, min(requested, len(pool)))
            selected.extend(picked)
            carry = requested - len(picked)
            if carry:
                shortfalls.append({"category": category, "band": band, "requested": requested, "selected": len(picked)})
    return selected, shortfalls


def assign_splits(selected: Sequence[ProductStats], dev_fraction: float, seed: int) -> dict[str, str]:
    """Stratified dev/test split by (category, band)."""
    groups: dict[tuple[str, str], list[ProductStats]] = {}
    for s in selected:
        groups.setdefault((s.category, band_for(s.review_count)), []).append(s)
    splits: dict[str, str] = {}
    for (category, band), members in groups.items():
        members = sorted(members, key=lambda s: s.product_id)
        random.Random(f"{seed}:{category}:{band}").shuffle(members)
        n_dev = math.floor(len(members) * dev_fraction + 0.5)
        for i, s in enumerate(members):
            splits[s.product_id] = "dev" if i < n_dev else "test"
    return splits


def build_manifest(
    products_path: Path,
    questions_path: Path,
    selected: Sequence[ProductStats],
    splits: Mapping[str, str],
    shortfalls: list[dict],
    rules: Mapping,
) -> dict:
    return {
        "manifest_version": 1,
        "dataset_id": DATASET_ID,
        "source": {
            "products_file": str(products_path),
            "products_sha256": file_sha256(products_path),
            "questions_file": str(questions_path),
            "questions_sha256": file_sha256(questions_path),
        },
        "rules": dict(rules),
        "products": [
            {
                "product_id": s.product_id,
                "category": s.category,
                "band": band_for(s.review_count),
                "review_count": s.review_count,
                "question_count": s.question_count,
                "split": splits[s.product_id],
            }
            for s in selected
        ],
        "shortfalls": shortfalls,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ReviewProof sampling manifest.")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rules = {
        "categories": list(DEFAULT_CATEGORIES),
        "per_band": dict(DEFAULT_PER_BAND),
        "min_questions": DEFAULT_MIN_QUESTIONS,
        "seed": DEFAULT_SEED,
        "dev_fraction": DEFAULT_DEV_FRACTION,
        "bands": dict(BAND_RULES),
    }
    stats = scan_product_stats(args.products, args.questions)
    selected, shortfalls = select_sample(stats, DEFAULT_CATEGORIES, DEFAULT_PER_BAND, DEFAULT_MIN_QUESTIONS, DEFAULT_SEED)
    splits = assign_splits(selected, DEFAULT_DEV_FRACTION, DEFAULT_SEED)
    manifest = build_manifest(args.products, args.questions, selected, splits, shortfalls, rules)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(selected)} products, shortfalls={shortfalls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
