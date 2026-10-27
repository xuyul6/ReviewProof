from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_device: str | None = None  # None: CUDA if available, else CPU


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
    return Settings(
        database_url=env.get(
            "REVIEWPROOF_DATABASE_URL",
            "postgresql://reviewproof:reviewproof@localhost:54329/reviewproof",
        ),
        embedding_model=env.get("REVIEWPROOF_EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_dim=_int(env, "REVIEWPROOF_EMBEDDING_DIM", 1024),
        embedding_batch_size=_int(env, "REVIEWPROOF_EMBEDDING_BATCH_SIZE", 32),
        embedding_device=env.get("REVIEWPROOF_EMBEDDING_DEVICE", "").strip() or None,
    )
