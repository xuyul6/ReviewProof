"""Ingest the sampled dataset: verify sources, load Postgres, embed, verify.

Run as `python -m reviewproof.ingest --manifest samples/manifest_v1.json`.
Re-running is safe: rows are upserted and only missing embeddings are computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import psycopg

from reviewproof.config import Settings, load_settings
from reviewproof.db import EMBEDDING_DIM, apply_schema, connect, embedding_column_dim
from reviewproof.embedding import Embedder, SentenceTransformerEmbedder
from reviewproof.ingest.amazonqa import file_sha256, load_products, load_questions
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset


def default_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size, settings.embedding_device
    )


def progress_printer(every: int = 20) -> Callable[[int, int], None]:
    """Report embedding progress on stderr every `every` batches and at the end; stdout stays JSON."""
    batches = 0

    def report(done: int, total: int) -> None:
        nonlocal batches
        batches += 1
        if batches % every == 0 or done == total:
            print(f"embedded {done:,}/{total:,} pending chunks", file=sys.stderr, flush=True)

    return report


def verify_ingest(conn: psycopg.Connection, manifest: Mapping, require_embeddings: bool, embedding_model: str) -> dict:
    """Counts chunks with no vector, or a vector from a model other than embedding_model, as missing."""
    ids = [p["product_id"] for p in manifest["products"]]
    present = {r[0] for r in conn.execute("SELECT product_id FROM products WHERE product_id = ANY(%s)", (ids,))}

    def count(sql: str, *extra) -> int:
        return conn.execute(sql, (ids, *extra)).fetchone()[0]

    report = {
        "products": len(present),
        "missing_products": [pid for pid in ids if pid not in present],
        "reviews": count("SELECT count(*) FROM reviews WHERE product_id = ANY(%s)"),
        "chunks": count("SELECT count(*) FROM chunks WHERE product_id = ANY(%s)"),
        "chunks_without_embedding": count(
            "SELECT count(*) FROM chunks WHERE product_id = ANY(%s)"
            " AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %s)", embedding_model),
        "questions": count("SELECT count(*) FROM questions WHERE product_id = ANY(%s)"),
    }
    conn.rollback()
    report["ok"] = not report["missing_products"] and (
        not require_embeddings or report["chunks_without_embedding"] == 0)
    return report


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Load the sampled AmazonQA dataset into Postgres.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--products-file", type=Path, help="defaults to the manifest's source.products_file")
    parser.add_argument("--questions-file", type=Path, help="defaults to the manifest's source.questions_file")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    if settings.embedding_dim != EMBEDDING_DIM:
        print(f"REVIEWPROOF_EMBEDDING_DIM={settings.embedding_dim} does not match the schema's "
              f"vector({EMBEDDING_DIM})", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = manifest["source"]
    products_file = args.products_file or Path(source["products_file"])
    questions_file = args.questions_file or Path(source["questions_file"])
    for path, expected in ((products_file, source["products_sha256"]), (questions_file, source["questions_sha256"])):
        actual = file_sha256(path)
        if actual != expected:
            print(f"清单与本地文件不一致: {path}\n  expected sha256 {expected}\n  actual   sha256 {actual}",
                  file=sys.stderr)
            return 2

    ids = [p["product_id"] for p in manifest["products"]]
    products = load_products(products_file, ids)
    questions = load_questions(questions_file, ids)

    conn = connect(settings.database_url)
    try:
        apply_schema(conn)
        column_dim = embedding_column_dim(conn)
        if column_dim != EMBEDDING_DIM:
            print(f"chunks.embedding is vector({column_dim}) but the code expects vector({EMBEDDING_DIM})",
                  file=sys.stderr)
            return 2
        try:
            load_report = load_dataset(conn, manifest, products, questions)
        except psycopg.errors.ForeignKeyViolation as exc:
            # Gold annotations and saved QA runs pin their questions/products; the load was rolled back.
            print(f"导入已回滚：这次导入会删除仍被标注或问答记录引用的行（{exc.diag.constraint_name}）。\n"
                  f"  {exc.diag.message_detail or exc}\n"
                  "  已有的金标数据不能随重新导入一起丢掉；如果确实要换数据，请先导出并另建标注集。",
                  file=sys.stderr)
            return 2
        embedded = 0
        embedding_model = settings.embedding_model
        if not args.skip_embeddings:
            embedder = (embedder_factory or default_embedder)(settings)
            embedding_model = embedder.model_name
            embedded = embed_pending_chunks(
                conn, embedder, settings.embedding_batch_size, product_ids=ids, on_batch=progress_printer())
        verify = verify_ingest(conn, manifest, require_embeddings=not args.skip_embeddings,
                               embedding_model=embedding_model)
    finally:
        conn.close()

    print(json.dumps({"load": asdict(load_report), "embedded": embedded, "verify": verify}, indent=2))
    return 0 if verify["ok"] else 1
