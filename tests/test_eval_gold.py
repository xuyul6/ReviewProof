import json
import tempfile
import unittest
from pathlib import Path

from reviewproof.eval.gold import GoldError, GoldLabel, agreement, gold_qrels, load_gold
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, make_manifest, make_product


def row(qid, answerability, grades, **overrides):
    base = {"label_set": "g", "guideline_version": "gold-v1", "dataset_id": "test-dataset", "qid": qid,
            "product_id": "A", "split": "dev", "question_type": "descriptive", "answerability": answerability,
            "verdict": None, "grades": grades, "rationale": "r", "annotator": "human", "pool": list(grades),
            "draft": None, "blind": None, "changed_from_draft": True}
    return {**base, **overrides}


def label(qid, answerability, grades=None, draft=None, blind=None, changed=False):
    return GoldLabel(qid=qid, split="dev", product_id="A", question_type="descriptive", answerability=answerability,
                     verdict=None, grades=grades or {}, draft=draft, blind=blind, changed_from_draft=changed)


class LoadGoldTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        q1 = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="descriptive",
                      is_answerable=True, answers=(), review_snippets=())
        load_dataset(self.conn, make_manifest("A"), {"A": make_product("A")}, {"A": [q1]})
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "labels.jsonl"

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def write(self, *rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return self.path

    def test_loads_labels(self):
        labels = load_gold(self.conn, self.write(row("q1", "answered", {"A-r00001-c00": 2, "A-r00000-c00": 0})))
        self.assertEqual(labels["q1"].grades, {"A-r00001-c00": 2, "A-r00000-c00": 0})

    def test_load_gold_rejects_other_dataset_and_missing_chunks(self):
        for bad in (row("q1", "answered", {"A-r00001-c00": 2}, dataset_id="other"),
                    row("q1", "answered", {"A-r09999-c00": 2}),
                    row("q1", "answered", {"B-r00000-c00": 2}),
                    row("q2", "answered", {"A-r00001-c00": 2})):
            with self.subTest(bad=bad), self.assertRaises(GoldError):
                load_gold(self.conn, self.write(bad))
        with self.assertRaises(GoldError):
            load_gold(self.conn, Path(self._tmp.name) / "missing.jsonl")


class GoldPureTests(unittest.TestCase):
    def test_gold_qrels_only_answerable_and_grade_two_is_relevant(self):
        labels = {"a": label("a", "answered", {"c1": 2, "c2": 1, "c3": 0}), "p": label("p", "partial", {"c4": 2}),
                  "u": label("u", "unanswerable", {"c5": 1})}
        qrels = gold_qrels(labels)
        self.assertEqual(set(qrels), {"a", "p"})
        self.assertEqual(qrels["a"], {"c1": 2, "c2": 1, "c3": 0})

    def test_agreement(self):
        labels = {
            "a": label("a", "answered", draft={"answerability": "answered"}, blind={"answerability": "answered"}),
            "b": label("b", "unanswerable", draft={"answerability": "answered"}, changed=True,
                       blind={"answerability": "unanswerable"}),
            "c": label("c", "partial", draft={"answerability": "partial"}),
        }
        result = agreement(labels, {"a": True, "b": True, "c": False})
        self.assertEqual(result["gold_vs_silver"]["n"], 3)
        self.assertAlmostEqual(result["gold_vs_silver"]["agreement"], 1 / 3)
        self.assertEqual(result["draft_vs_final"]["n"], 3)
        self.assertAlmostEqual(result["draft_vs_final"]["changed_share"], 1 / 3)
        self.assertEqual(result["blind_vs_draft"]["n"], 2)
        self.assertAlmostEqual(result["blind_vs_draft"]["agreement"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
