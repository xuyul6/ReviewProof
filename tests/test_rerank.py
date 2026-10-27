import types
import unittest
from unittest.mock import MagicMock, patch

import torch  # noqa: F401  imported before patch.dict(sys.modules) so the patch never unloads it

from reviewproof.retrieval.rerank import CrossEncoderReranker
from tests.support import FakeReranker


def fake_cross_encoder_module(scores_per_pair=0.5):
    model = MagicMock()
    model.predict.side_effect = lambda pairs, **kwargs: [scores_per_pair] * len(pairs)
    return types.SimpleNamespace(CrossEncoder=MagicMock(return_value=model)), model


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_lazy_load_device_and_pairs(self):
        module, model = fake_cross_encoder_module()
        with patch.dict("sys.modules", {"sentence_transformers": module}), \
                patch("torch.cuda.is_available", return_value=False):
            reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
            module.CrossEncoder.assert_not_called()
            self.assertEqual(reranker.score("q", ["a", "b"]), [0.5, 0.5])
        module.CrossEncoder.assert_called_once_with("BAAI/bge-reranker-v2-m3", device="cpu", max_length=512)
        self.assertEqual(model.predict.call_args.args[0], [("q", "a"), ("q", "b")])
        self.assertFalse(model.predict.call_args.kwargs["show_progress_bar"])

    def test_explicit_device(self):
        module, _ = fake_cross_encoder_module()
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            CrossEncoderReranker("m", device="mps").score("q", ["a"])
        module.CrossEncoder.assert_called_once_with("m", device="mps", max_length=512)

    def test_empty_texts_do_not_load_model(self):
        module, _ = fake_cross_encoder_module()
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            self.assertEqual(CrossEncoderReranker("m").score("q", []), [])
        module.CrossEncoder.assert_not_called()

    def test_wrong_score_count_raises(self):
        module, model = fake_cross_encoder_module()
        model.predict.side_effect = lambda pairs, **kwargs: [0.1]
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            with self.assertRaises(ValueError):
                CrossEncoderReranker("m", device="cpu").score("q", ["a", "b"])


class FakeRerankerTests(unittest.TestCase):
    def test_scores_shared_words(self):
        self.assertEqual(FakeReranker().score("is it loud", ["Too loud for me.", "Cheap plastic."]), [1.0, 0.0])

    def test_fail(self):
        with self.assertRaises(RuntimeError):
            FakeReranker(fail=True).score("q", ["a"])


if __name__ == "__main__":
    unittest.main()
