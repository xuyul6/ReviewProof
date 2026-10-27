import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reviewproof.annotate.cli import main
from reviewproof.annotate.pool import search_product
from reviewproof.annotate.store import Label, get_item, list_items, save_label
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.retrieval.bm25 import BM25Cache
from tests.support import (TEST_DATABASE_URL, DatabaseTestCase, FakeEmbedder, FakeReranker, make_manifest,
                           make_product)

NOISE = [f"Noise review {i}: the fan is loud at night." for i in range(6)] + ["Quiet enough for an office."]


def question(qid, product, qtype="descriptive", answerable=True):
    return Question(qid=qid, product_id=product, question_text="Is the fan loud?", question_type=qtype,
                    is_answerable=answerable, answers=({"answerText": "SECRET-ANSWER"},),
                    review_snippets=("SECRET-SNIPPET",))


class AnnotateCliTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        manifest = make_manifest("A", "C")
        manifest["products"][1]["split"] = "test"
        load_dataset(self.conn, manifest, {"A": make_product("A"), "C": make_product("C", NOISE)},
                     {"A": [question("qa", "A", "yesno"), question("qa2", "A", answerable=False)],
                      "C": [question("qc", "C"), question("qc2", "C", "yesno")]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=50)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def run_main(self, argv):
        env = {"REVIEWPROOF_DATABASE_URL": TEST_DATABASE_URL, "REVIEWPROOF_FULL_CONTEXT_MAX_TOKENS": "20"}
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, embedder_factory=lambda s: FakeEmbedder(), reranker_factory=lambda s: FakeReranker())
        return code, out.getvalue(), err.getvalue()

    def pick(self):
        code, _, err = self.run_main(["pick", "--set", "g", "--dev", "2", "--test", "2", "--blind-fraction", "0.5",
                                      "--out-root", str(self.root)])
        self.assertEqual(code, 0, err)

    def chunk_ids(self, product):
        rows = self.conn.execute("SELECT chunk_id FROM chunks WHERE product_id = %s ORDER BY chunk_id",
                                 (product,)).fetchall()
        self.conn.rollback()
        return [r[0] for r in rows]

    def test_pick_writes_items_with_pools_and_blind_share(self):
        self.pick()
        rows = [json.loads(l) for l in (self.root / "g" / "items.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["qid"] for r in rows), ["qa", "qa2", "qc", "qc2"])
        by_qid = {r["qid"]: r for r in rows}
        self.assertEqual(by_qid["qa"]["pool"], self.chunk_ids("A"))  # full-context product: every chunk
        self.assertEqual(by_qid["qc"]["pool"], sorted(by_qid["qc"]["pool"]))
        self.assertTrue(set(by_qid["qc"]["pool"]) <= set(self.chunk_ids("C")) and by_qid["qc"]["pool"])
        for split in ("dev", "test"):
            self.assertEqual(sum(r["blind"] for r in rows if r["split"] == split), 1)
        for key in ("dataset_id", "pool_depth", "retrieval_params", "git_commit"):
            self.assertIn(key, rows[0])
        self.assertEqual(len(list_items(self.conn, "g")), 4)

    def test_pick_twice_exits_2(self):
        self.pick()
        code, _, _ = self.run_main(["pick", "--set", "g", "--dev", "1", "--test", "1", "--out-root", str(self.root)])
        self.assertEqual(code, 2)

    def test_packet_has_question_and_pool_but_no_silver_snippet_or_answer(self):
        self.pick()
        code, out, err = self.run_main(["packet", "--set", "g", "--qids", "qa,qc"])
        self.assertEqual(code, 0, err)
        self.assertIn("Is the fan loud?", out)
        for cid in get_item(self.conn, "g", "qc")["pool"]:
            self.assertIn(f"[{cid['chunk_id']}]", out)
        for secret in ("SECRET-SNIPPET", "SECRET-ANSWER", "is_answerable", "silver"):
            self.assertNotIn(secret, out)

    def test_import_drafts_reports_rejects_and_keeps_good_lines(self):
        self.pick()
        pool_a = [c["chunk_id"] for c in get_item(self.conn, "g", "qa")["pool"]]
        good = {"qid": "qa", "answerability": "answered", "verdict": "yes",
                "grades": dict({c: 0 for c in pool_a}, **{pool_a[1]: 2}), "rationale": "r1", "annotator": "agent:t"}
        bad = {"qid": "qc", "answerability": "answered", "verdict": None, "grades": {"A-r00000-c00": 2},
               "rationale": "x", "annotator": "agent:t"}
        path = self.root / "drafts.jsonl"
        path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
        code, out, err = self.run_main(["import-drafts", "--set", "g", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("qc:", out + err)
        self.assertEqual(get_item(self.conn, "g", "qa")["labels"]["draft"]["source"], "draft")
        self.assertEqual(get_item(self.conn, "g", "qc")["labels"], {})

    def test_import_drafts_survives_malformed_shapes(self):
        self.pick()
        pool_a = [c["chunk_id"] for c in get_item(self.conn, "g", "qa")["pool"]]
        grades = dict({c: 0 for c in pool_a}, **{pool_a[1]: 2})
        base = {"answerability": "answered", "verdict": "yes", "rationale": "r", "annotator": "agent:t"}
        path = self.root / "shapes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in (
            dict(base, qid="qa", grades=list(grades)),
            dict(base, qid="qa", grades=grades, rationale=None),
            dict(base, qid="qa", grades=dict(grades, **{pool_a[0]: True})),
            "not an object",
        )) + "\n" + '{"qid": 12345, "answerability": "answered"}\n')
        code, out, err = self.run_main(["import-drafts", "--set", "g", str(path)])
        self.assertEqual(code, 1, err)
        self.assertIn("accepted 0, rejected 5", out)
        self.assertEqual(get_item(self.conn, "g", "qa")["labels"], {})

    def test_search_excludes_pool_and_other_products(self):
        hits = search_product(self.conn, "C", "office", BM25Cache(), exclude={"C-r00000-c00"})
        self.assertEqual([h["chunk_id"] for h in hits], ["C-r00006-c00"])
        hits = search_product(self.conn, "C", "loud", BM25Cache(), exclude={"C-r00000-c00"})
        self.assertTrue(hits and all(h["chunk_id"].startswith("C-") for h in hits))
        self.assertNotIn("C-r00000-c00", [h["chunk_id"] for h in hits])
        self.assertEqual(set(hits[0]), {"chunk_id", "review_id", "text"})

    def test_search_command_lists_out_of_pool_chunks(self):
        self.pick()
        pool = {c["chunk_id"] for c in get_item(self.conn, "g", "qc")["pool"]}
        code, out, err = self.run_main(["search", "--set", "g", "--qid", "qc", "--query", "fan"])
        self.assertEqual(code, 0, err)
        self.assertFalse(any(f"[{cid}]" in out for cid in pool))

    def test_export_writes_only_final_labels(self):
        self.pick()
        pool_a = [c["chunk_id"] for c in get_item(self.conn, "g", "qa")["pool"]]
        grades = dict({c: 0 for c in pool_a}, **{pool_a[1]: 2})
        for source in ("draft", "blind", "final"):
            if source == "blind" and not get_item(self.conn, "g", "qa")["blind"]:
                continue
            save_label(self.conn, "g", Label("qa", source, "answered", "yes", grades if source != "blind" else {},
                                             "r1", "human"))
        code, _, err = self.run_main(["export", "--set", "g", "--out-root", str(self.root)])
        self.assertEqual(code, 0, err)
        rows = [json.loads(l) for l in (self.root / "g" / "labels.jsonl").read_text().splitlines()]
        self.assertEqual([r["qid"] for r in rows], ["qa"])

    def test_status_counts(self):
        self.pick()
        code, out, _ = self.run_main(["status", "--set", "g"])
        self.assertEqual(code, 0)
        self.assertIn("pending", out)


if __name__ == "__main__":
    unittest.main()
