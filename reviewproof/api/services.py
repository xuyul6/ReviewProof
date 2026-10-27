"""Everything a request needs, built once per process: settings, the LLM and loaded models."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import psycopg

from reviewproof.config import Settings
from reviewproof.db import apply_schema, connect
from reviewproof.llm import LLM
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.cli import default_embedder, default_reranker
from reviewproof.workflow.cli import default_llm
from reviewproof.workflow.state import RetrievalDeps


@dataclass
class Services:
    settings: Settings
    llm: LLM
    retrieval: RetrievalDeps
    # One workflow at a time: BM25Cache is not thread-safe and CPU inference gains nothing from overlap.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def connect(self) -> psycopg.Connection:
        return connect(self.settings.database_url)


def build_services(settings: Settings) -> Services:
    conn = connect(settings.database_url)
    try:
        apply_schema(conn)  # creates qa_runs on databases loaded before phase 4
    finally:
        conn.close()
    embedder = default_embedder(settings)
    retrieval = RetrievalDeps(
        embedder=embedder, embedding_model=embedder.model_name, reranker=default_reranker(settings),
        bm25_cache=BM25Cache(), params=RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens))
    return Services(settings=settings, llm=default_llm(settings), retrieval=retrieval)


def warm_up(services: Services) -> None:
    """Load the embedding and rerank models now instead of on the first question."""
    services.retrieval.embedder.embed_query("warm up")
    if services.retrieval.reranker is not None:
        services.retrieval.reranker.score("warm up", ["warm up"])
