"""Read-only MCP server: lets an external LLM client list products, search one product's reviews
and read a review. No QA tool: the calling client is itself an LLM and only needs the evidence.
Community answers, snippets and answerability labels are never returned.

Run as `python -m reviewproof.mcp_server` (REVIEWPROOF_MCP_TRANSPORT: stdio, sse or streamable-http).
"""
from __future__ import annotations

import os
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from reviewproof.api import queries
from reviewproof.api.services import Services, build_services, warm_up
from reviewproof.retrieval import ProductNotFoundError, retrieve


def tools(services: Services) -> dict[str, Callable[..., Any]]:
    def list_products(category: str | None = None) -> list[dict]:
        """List the sampled products (ASIN, category, review count, retrieval strategy)."""
        with services.connect() as conn:
            return queries.list_products(conn, full_context_max_tokens=services.settings.full_context_max_tokens,
                                         category=category)["products"]

    def search_reviews(product_id: str, query: str) -> dict:
        """Search one product's review chunks for a question; returns ranked evidence."""
        deps = services.retrieval
        with services.lock, services.connect() as conn:
            try:
                r = retrieve(conn, product_id, query, embedder=deps.embedder, embedding_model=deps.embedding_model,
                             reranker=deps.reranker, bm25_cache=deps.bm25_cache, params=deps.params, strategy="auto")
            except ProductNotFoundError:
                return {"error": "product not found"}
        return {"strategy": r.strategy, "mode": r.mode, "service_status": r.service_status,
                "results": [{"chunk_id": c.chunk.chunk_id, "review_id": c.chunk.review_id, "text": c.chunk.text,
                             "dense_rank": c.dense_rank, "bm25_rank": c.bm25_rank, "rerank_score": c.rerank_score}
                            for c in r.candidates]}

    def get_review(review_id: str) -> dict:
        """Read one review's full text (review IDs are system-generated, not Amazon review IDs)."""
        with services.connect() as conn:
            body = queries.get_review(conn, review_id)
        return body if body is not None else {"error": "review not found in the current data"}

    return {"list_products": list_products, "search_reviews": search_reviews, "get_review": get_review}


def build_server(services: Services) -> MCPServer:
    server = MCPServer(name="reviewproof")
    for name, fn in tools(services).items():
        server.add_tool(fn, name=name)
    return server


def main() -> None:
    from reviewproof.config import load_settings

    services = build_services(load_settings())
    warm_up(services)
    build_server(services).run(transport=os.environ.get("REVIEWPROOF_MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
