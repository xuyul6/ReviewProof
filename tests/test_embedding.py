import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch  # noqa: F401  imported before patch.dict(sys.modules) so the patch never unloads it

from reviewproof.embedding import SentenceTransformerEmbedder


def fake_sentence_transformers(dim):
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), dim), dtype=np.float32)
    module = types.SimpleNamespace(SentenceTransformer=MagicMock(return_value=model))
    return module, model


class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_passes_normalize_and_loads_model_once(self):
        module, model = fake_sentence_transformers(1024)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            embedder = SentenceTransformerEmbedder("BAAI/bge-m3", 1024, batch_size=4)
            module.SentenceTransformer.assert_not_called()  # lazy
            self.assertEqual(len(embedder.embed_documents(["a", "b"])), 2)
            self.assertEqual(len(embedder.embed_query("q")), 1024)
        self.assertEqual(module.SentenceTransformer.call_args.args, ("BAAI/bge-m3",))
        for call in model.encode.call_args_list:
            self.assertTrue(call.kwargs["normalize_embeddings"])
            self.assertEqual(call.kwargs["batch_size"], 4)

    def test_defaults_to_cpu_even_when_mps_is_available(self):
        # MPS keeps a compiled graph per batch shape that empty_cache() cannot free: BGE-M3 grew
        # ~0.2 GiB per varied batch and ran out of memory on a 16 GB Mac; the CPU was as fast.
        module, _ = fake_sentence_transformers(1024)
        with patch.dict("sys.modules", {"sentence_transformers": module}), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("torch.backends.mps.is_available", return_value=True):
            SentenceTransformerEmbedder("m", 1024, batch_size=4).embed_documents(["a"])
        module.SentenceTransformer.assert_called_once_with("m", device="cpu")

    def test_prefers_cuda_when_available(self):
        module, _ = fake_sentence_transformers(1024)
        with patch.dict("sys.modules", {"sentence_transformers": module}), \
                patch("torch.cuda.is_available", return_value=True):
            SentenceTransformerEmbedder("m", 1024, batch_size=4).embed_documents(["a"])
        module.SentenceTransformer.assert_called_once_with("m", device="cuda")

    def test_explicit_device_wins(self):
        module, _ = fake_sentence_transformers(1024)
        with patch.dict("sys.modules", {"sentence_transformers": module}), \
                patch("torch.cuda.is_available", return_value=True):
            SentenceTransformerEmbedder("m", 1024, batch_size=4, device="mps").embed_documents(["a"])
        module.SentenceTransformer.assert_called_once_with("m", device="mps")

    def test_wrong_output_dimension_raises(self):
        module, _ = fake_sentence_transformers(8)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            embedder = SentenceTransformerEmbedder("m", 1024, batch_size=4)
            with self.assertRaisesRegex(ValueError, "1024"):
                embedder.embed_documents(["a"])


if __name__ == "__main__":
    unittest.main()
