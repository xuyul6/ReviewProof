from __future__ import annotations

from typing import Protocol, Sequence


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def default_device() -> str:
    """CUDA when present, otherwise CPU. Apple MPS is skipped on purpose: it keeps a compiled graph
    for every new batch shape and empty_cache() does not free them, so a BGE-M3 run grew ~0.2 GiB
    per batch and ran out of memory on a 16 GB Mac, while the CPU embedded just as fast."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (BGE-M3 by default). Loads on first use.

    BGE-M3 needs no query instruction prefix, so queries and documents share one path.
    """

    def __init__(self, model_name: str, dim: int, batch_size: int, device: str | None = None):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device or default_device())
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"{self.model_name} produced {vectors.shape[1]}-dim vectors, expected {self.dim}")
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
