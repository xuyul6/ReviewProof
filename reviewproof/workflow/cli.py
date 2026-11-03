"""Answer one question about one product from its reviews.

Run as `python -m reviewproof.workflow --product ASIN --question "..."` or `--qid QID`.
With --qid only the question text and product are read; community answers, snippets and
labels are never loaded or printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Callable, Sequence

from reviewproof.config import Settings, load_settings
from reviewproof.db import connect
from reviewproof.embedding import Embedder
from reviewproof.llm import LLM, ClaudeLLM
from reviewproof.retrieval import ProductNotFoundError, RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.cli import default_embedder, default_reranker
from reviewproof.retrieval.rerank import Reranker
from reviewproof.workflow.graph import QAResult, answer_question
from reviewproof.workflow.state import RetrievalDeps


def default_llm(settings: Settings) -> LLM:
    return ClaudeLLM(settings.llm_model, settings.llm_effort)


def format_result(r: QAResult) -> str:
    lines = [r.answer_text, ""]
    if r.note:
        lines += [r.note.heading, r.note.text, ""]
    lines.append(f"status={r.status}  service_status={r.service_status}  verdict={r.verdict}  route={r.route}")
    if r.citations:
        lines += ["", "citations:"]
        for c in r.citations:
            text = c.text if len(c.text) <= 160 else c.text[:159] + "…"
            lines.append(f"  [{c.number}] {c.chunk_id}  {text}")
    lines += ["", "trace:"]
    lines += [f"  {t.node:<8} {t.status:<8} {t.latency_ms:9.1f} ms  {t.detail}" for t in r.trace]
    lines += ["", f"llm_calls={r.llm_calls}  input_tokens={r.input_tokens}  output_tokens={r.output_tokens}  "
                  f"latency={r.latency_ms / 1000:.1f} s"]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    llm_factory: Callable[[Settings], LLM] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
    reranker_factory: Callable[[Settings], Reranker] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Answer a question about one product from its reviews.")
    parser.add_argument("--product", help="product ASIN (optional with --qid)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--qid", help="take the question text and product from the questions table")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.question is not None and not args.product:
        parser.error("--question needs --product")
    if args.question is not None and not args.question.strip():
        print("question is empty", file=sys.stderr)
        return 2

    settings = load_settings()
    conn = connect(settings.database_url)
    try:
        product_id, question = args.product, args.question
        if args.qid is not None:
            row = conn.execute("SELECT product_id, question_text FROM questions WHERE qid = %s", (args.qid,)).fetchone()
            conn.rollback()
            if row is None:
                print(f"question {args.qid} not found", file=sys.stderr)
                return 2
            if product_id and product_id != row[0]:
                print(f"question {args.qid} belongs to {row[0]}, not {product_id}", file=sys.stderr)
                return 2
            product_id, question = row
        embedder = (embedder_factory or default_embedder)(settings)
        deps = RetrievalDeps(
            embedder=embedder, embedding_model=embedder.model_name,
            reranker=None if args.no_rerank else (reranker_factory or default_reranker)(settings),
            bm25_cache=BM25Cache(),
            params=RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens))
        try:
            result = answer_question(conn, product_id, question, llm=(llm_factory or default_llm)(settings),
                                     retrieval=deps, max_rewrites=settings.max_rewrites, advise=True)
        except ProductNotFoundError:
            print(f"product {product_id} not found", file=sys.stderr)
            return 2
    finally:
        conn.close()

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else format_result(result))
    return 1 if result.service_status == "failed" else 0
