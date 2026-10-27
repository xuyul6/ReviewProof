import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewproof.ingest.amazonqa import file_sha256
from reviewproof.config import load_settings
from reviewproof.ingest.cli import default_embedder, main, verify_ingest
from tests.support import TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, write_jsonl

# A URL nothing listens on: any connection attempt raises, proving the command stopped earlier.
UNREACHABLE_DB = "postgresql://nobody:x@127.0.0.1:1/none?connect_timeout=1"


class IngestFixture:
    def make_files(self, tmp: Path, wrong_sha: bool = False) -> Path:
        products = write_jsonl(tmp / "products.jsonl", [
            {"asin": "A", "category": "Electronics", "questions": [],
             "reviews": [{"reviewText": "Battery lasts two days.", "helpful": [1, 1]},
                         {"reviewText": "Too loud.", "helpful": [0, 0]}]},
            {"asin": "B", "category": "Beauty", "questions": [],
             "reviews": [{"reviewText": "Smells nice.", "helpful": [0, 0]}]},
        ])
        questions = write_jsonl(tmp / "questions.jsonl", [
            {"qid": 1, "asin": "A", "questionText": "Is it loud?", "questionType": "yesno",
             "is_answerable": 1, "answers": [], "review_snippets": ["Too loud."]},
        ])
        self.manifest = {
            "manifest_version": 1,
            "dataset_id": "test-dataset",
            "source": {
                "products_file": str(products),
                "products_sha256": "0" * 64 if wrong_sha else file_sha256(products),
                "questions_file": str(questions),
                "questions_sha256": file_sha256(questions),
            },
            "rules": {},
            "products": [
                {"product_id": "A", "category": "Electronics", "band": "small", "review_count": 2,
                 "question_count": 1, "split": "dev"},
                {"product_id": "B", "category": "Beauty", "band": "small", "review_count": 1,
                 "question_count": 0, "split": "test"},
            ],
            "shortfalls": [],
        }
        path = tmp / "manifest.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return path

    def run_main(self, argv, env):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, embedder_factory=lambda s: FakeEmbedder())
        return code, out.getvalue(), err.getvalue()


class IngestCliOfflineTests(IngestFixture, unittest.TestCase):
    def test_sha_mismatch_exits_2_before_touching_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.make_files(Path(tmp), wrong_sha=True)
            code, _, err = self.run_main(["--manifest", str(manifest)], {"REVIEWPROOF_DATABASE_URL": UNREACHABLE_DB})
        self.assertEqual(code, 2)
        self.assertIn("清单与本地文件不一致", err)
        self.assertIn("0" * 64, err)

    def test_default_embedder_uses_configured_device(self):
        settings = load_settings({"REVIEWPROOF_EMBEDDING_DEVICE": "mps"})
        self.assertEqual(default_embedder(settings).device, "mps")

    def test_embedding_dim_mismatch_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.make_files(Path(tmp))
            code, _, err = self.run_main(
                ["--manifest", str(manifest)],
                {"REVIEWPROOF_DATABASE_URL": UNREACHABLE_DB, "REVIEWPROOF_EMBEDDING_DIM": "8"})
        self.assertEqual(code, 2)
        self.assertIn("REVIEWPROOF_EMBEDDING_DIM", err)


class IngestCliDatabaseTests(IngestFixture, DatabaseTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest_path = self.make_files(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_end_to_end_with_fake_embedder(self):
        code, out, _ = self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
        self.assertEqual(code, 0)
        summary = json.loads(out)
        self.assertEqual(summary["load"], {"products": 2, "reviews": 3, "chunks": 3, "questions": 1})
        self.assertEqual(summary["embedded"], 3)
        report = verify_ingest(self.conn, self.manifest, require_embeddings=True)
        self.assertEqual(report, {"products": 2, "missing_products": [], "reviews": 3, "chunks": 3,
                                  "chunks_without_embedding": 0, "questions": 1, "ok": True})

    def test_progress_goes_to_stderr_and_stdout_stays_json(self):
        code, out, err = self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
        self.assertEqual(code, 0)
        self.assertIn("embedded 3/3 pending chunks", err)
        self.assertEqual(json.loads(out)["embedded"], 3)

    def test_second_run_embeds_nothing(self):
        self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
        code, out, _ = self.run_main(["--manifest", str(self.manifest_path)], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
        self.assertEqual((code, json.loads(out)["embedded"]), (0, 0))

    def test_skip_embeddings_still_ok(self):
        code, out, _ = self.run_main(
            ["--manifest", str(self.manifest_path), "--skip-embeddings"], {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["verify"]["chunks_without_embedding"], 3)

    def test_verify_reports_missing_products(self):
        report = verify_ingest(self.conn, self.manifest, require_embeddings=False)
        self.assertEqual((report["missing_products"], report["ok"]), (["A", "B"], False))


if __name__ == "__main__":
    unittest.main()
