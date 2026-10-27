from __future__ import annotations

from typing import Protocol, Sequence

from reviewproof.embedding import default_device


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Local sentence-transformers cross-encoder (bge-reranker-v2-m3 by default). Loads on first use."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device or default_device(), max_length=self.max_length)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._load().predict(
            [(query, text) for text in texts], batch_size=self.batch_size, show_progress_bar=False)
        if len(scores) != len(texts):
            raise ValueError(f"{self.model_name} returned {len(scores)} scores for {len(texts)} texts")
        return [float(s) for s in scores]
