import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval.cli import main
from tests.support import (TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, FakeReranker, make_manifest,
                           make_product)

QUESTION = Question(
    qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno", is_answerable=True,
    answers=({"answerText": "SECRET-ANSWER"},), review_snippets=("SECRET-SNIPPET",))


class RetrievalCliTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])},
                     {"A": [QUESTION]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)

    def run_main(self, argv, embedder=None):
        out, err = io.StringIO(), io.StringIO()
        env = {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL, "REVIEWPROOF_EMBEDDING_MODEL": "fake"}
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, embedder_factory=lambda s: embedder or FakeEmbedder(),
                        reranker_factory=lambda s: FakeReranker())
        return code, out.getvalue(), err.getvalue()

    def test_query_prints_table(self):
        code, out, _ = self.run_main(["--product", "A", "--query", "Too loud for me.", "--strategy", "hybrid"])
        self.assertEqual(code, 0)
        self.assertIn("dense+bm25+rerank", out)
        self.assertIn("A-r00001-c00", out)

    def test_qid_reads_question_and_never_prints_answers(self):
        code, out, _ = self.run_main(["--qid", "q1", "--json"])
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual((result["product_id"], result["query"]), ("A", "Is it loud?"))
        self.assertNotIn("SECRET", out)

    def test_no_rerank(self):
        code, out, _ = self.run_main(["--product", "A", "--query", "loud", "--strategy", "hybrid", "--no-rerank", "--json"])
        self.assertEqual((code, json.loads(out)["mode"]), (0, "dense+bm25"))

    def test_qid_product_mismatch_exits_2(self):
        self.assertEqual(self.run_main(["--qid", "q1", "--product", "B"])[0], 2)

    def test_unknown_product_or_qid_exits_2(self):
        self.assertEqual(self.run_main(["--product", "ZZZ", "--query", "loud"])[0], 2)
        self.assertEqual(self.run_main(["--qid", "nope"])[0], 2)

    def test_service_failure_exits_1(self):
        code, out, _ = self.run_main(["--product", "A", "--query", "声音大吗？", "--strategy", "hybrid"],
                                     embedder=FakeEmbedder(fail_queries=True))
        self.assertEqual(code, 1)
        self.assertIn("failed", out)


if __name__ == "__main__":
    unittest.main()
