"""Show how one product's chunks rank for one question.

Run as `python -m reviewproof.retrieval --product ASIN --query "..."` or `--qid QID`.
Only retrieval output is printed: never community answers, snippets or answerability labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Callable, Sequence

from reviewproof.config import Settings, load_settings
from reviewproof.db import connect
from reviewproof.embedding import Embedder, SentenceTransformerEmbedder
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.rerank import CrossEncoderReranker, Reranker
from reviewproof.retrieval.retriever import STRATEGIES, ProductNotFoundError, retrieve
from reviewproof.retrieval.types import RetrievalParams, RetrievalResult


def default_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size, settings.embedding_device)


def default_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings.rerank_model, settings.embedding_device)


def format_table(result: RetrievalResult) -> str:
    lines = [f"strategy={result.strategy}  mode={result.mode}  service_status={result.service_status}", ""]
    for c in result.channels:
        lines.append(f"  {c.name:<7} {c.status:<8} {c.latency_ms:8.1f} ms  hits={c.hits:<3} {c.reason or ''}")
    lines += ["", f"{'#':>2}  {'chunk_id':<26} {'dense':>5} {'bm25':>5} {'rrf':>7} {'rerank':>7}  text"]

    def fmt(value, spec, width):
        return format(value, spec) if value is not None else "-".rjust(width)

    for i, c in enumerate(result.candidates, start=1):
        text = c.chunk.text if len(c.chunk.text) <= 100 else c.chunk.text[:99] + "…"
        lines.append(f"{i:>2}  {c.chunk.chunk_id:<26} {fmt(c.dense_rank, '>5', 5)} {fmt(c.bm25_rank, '>5', 5)} "
                     f"{fmt(c.rrf_score, '>7.4f', 7)} {fmt(c.rerank_score, '>7.3f', 7)}  {text}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
    reranker_factory: Callable[[Settings], Reranker] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Rank one product's review chunks for a question.")
    parser.add_argument("--product", help="product ASIN (optional with --qid)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--qid", help="take the question text and product from the questions table")
    parser.add_argument("--strategy", choices=STRATEGIES, default="auto")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.query is not None and not args.product:
        parser.error("--query needs --product")

    settings = load_settings()
    conn = connect(settings.database_url)
    try:
        product_id, query = args.product, args.query
        if args.qid is not None:
            row = conn.execute("SELECT product_id, question_text FROM questions WHERE qid = %s", (args.qid,)).fetchone()
            conn.rollback()
            if row is None:
                print(f"question {args.qid} not found", file=sys.stderr)
                return 2
            if product_id and product_id != row[0]:
                print(f"question {args.qid} belongs to {row[0]}, not {product_id}", file=sys.stderr)
                return 2
            product_id, query = row
        embedder = (embedder_factory or default_embedder)(settings)
        reranker = None if args.no_rerank else (reranker_factory or default_reranker)(settings)
        try:
            result = retrieve(
                conn, product_id, query, embedder=embedder, embedding_model=embedder.model_name, reranker=reranker,
                bm25_cache=BM25Cache(), params=RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens),
                strategy=args.strategy)
        except ProductNotFoundError:
            print(f"product {product_id} not found", file=sys.stderr)
            return 2
    finally:
        conn.close()

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else format_table(result))
    return 1 if result.service_status == "failed" else 0
