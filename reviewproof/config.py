from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


LLM_EFFORTS = ("low", "medium", "high")


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_device: str | None = None  # None: CUDA if available, else CPU
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    full_context_max_tokens: int = 6000  # products at or under this use full context, not retrieval
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "low"
    max_rewrites: int = 1


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read settings from env (defaults to os.environ). Does not load .env files."""
    env = os.environ if env is None else env
    full_context_max_tokens = _int(env, "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS", 6000)
    if full_context_max_tokens < 0:
        raise ValueError(f"REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS must be >= 0, got {full_context_max_tokens}")
    llm_effort = env.get("REVIEWPROOF_LLM_EFFORT", "").strip() or "low"
    if llm_effort not in LLM_EFFORTS:
        raise ValueError(f"REVIEWPROOF_LLM_EFFORT must be one of {LLM_EFFORTS}, got {llm_effort!r}")
    max_rewrites = _int(env, "REVIEWPROOF_MAX_REWRITES", 1)
    if not 0 <= max_rewrites <= 2:
        raise ValueError(f"REVIEWPROOF_MAX_REWRITES must be between 0 and 2, got {max_rewrites}")
    return Settings(
        database_url=env.get(
            "REVIEWPROOF_DATABASE_URL",
            "postgresql://reviewproof:reviewproof@localhost:54329/reviewproof",
        ),
        embedding_model=env.get("REVIEWPROOF_EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_dim=_int(env, "REVIEWPROOF_EMBEDDING_DIM", 1024),
        embedding_batch_size=_int(env, "REVIEWPROOF_EMBEDDING_BATCH_SIZE", 32),
        embedding_device=env.get("REVIEWPROOF_EMBEDDING_DEVICE", "").strip() or None,
        rerank_model=env.get("REVIEWPROOF_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        full_context_max_tokens=full_context_max_tokens,
        llm_model=env.get("REVIEWPROOF_LLM_MODEL", "").strip() or "claude-haiku-5-5",
        llm_effort=llm_effort,
        max_rewrites=max_rewrites,
    )
