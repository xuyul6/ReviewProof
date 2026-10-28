import unittest

import psycopg

from reviewproof.annotate.store import (GUIDELINE_VERSION, ItemSpec, Label, LabelError, add_items, export_labels,
                                        get_item, list_items, save_label)
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, make_manifest, make_product

POOL = ("A-r00000-c00", "A-r00001-c00", "A-r00002-c00")
ALL_ZERO = {cid: 0 for cid in POOL}


def question(qid, qtype):
    return Question(qid=qid, product_id="A", question_text="Is it loud?", question_type=qtype, is_answerable=True,
                    answers=({"answerText": "SECRET-ANSWER"},), review_snippets=("SECRET-SNIPPET",))


def label(qid="qa", source="draft", answerability="answered", verdict="yes", grades=None, rationale="r1 says so",
          annotator="agent:test"):
    return Label(qid=qid, source=source, answerability=answerability, verdict=verdict,
                 grades=dict(ALL_ZERO, **{"A-r00001-c00": 2}) if grades is None else grades,
                 rationale=rationale, annotator=annotator)


class AnnotationStoreTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.questions = {"A": [question("qa", "yesno"), question("qd", "descriptive")]}
        load_dataset(self.conn, make_manifest("A", "B"), {"A": make_product("A"), "B": make_product("B")},
                     self.questions)
        add_items(self.conn, "gold_v1", [ItemSpec("qa", "dev", False, POOL), ItemSpec("qd", "dev", True, POOL)])

    def test_save_and_export_round_trip(self):
        save_label(self.conn, "gold_v1", label())
        save_label(self.conn, "gold_v1", label(source="final", annotator="human"))
        item = get_item(self.conn, "gold_v1", "qa")
        self.assertEqual([c["chunk_id"] for c in item["pool"]], list(POOL))
        self.assertEqual(item["pool"][1]["text"], "Too loud for me.")
        self.assertEqual(set(item["labels"]), {"draft", "final"})
        (row,) = export_labels(self.conn, "gold_v1")
        self.assertEqual((row["qid"], row["label_set"], row["dataset_id"], row["guideline_version"]),
                         ("qa", "gold_v1", "test-dataset", GUIDELINE_VERSION))
        self.assertEqual((row["answerability"], row["verdict"], row["annotator"]), ("answered", "yes", "human"))
        self.assertEqual(row["draft"]["grades"]["A-r00001-c00"], 2)
        self.assertIsNone(row["blind"])
        self.assertFalse(row["changed_from_draft"])
        self.assertEqual(row["pool"], list(POOL))

    def test_changed_from_draft_when_a_grade_differs(self):
        save_label(self.conn, "gold_v1", label())
        save_label(self.conn, "gold_v1", label(source="final", grades=dict(ALL_ZERO, **{"A-r00001-c00": 2,
                                                                                        "A-r00000-c00": 1})))
        self.assertTrue(export_labels(self.conn, "gold_v1")[0]["changed_from_draft"])

    def test_validation_rejects_and_writes_nothing(self):
        two = dict(ALL_ZERO, **{"A-r00001-c00": 2})
        bad = [
            label(grades=dict(two, **{"B-r00000-c00": 1})),             # chunk of another product
            label(grades=dict(two, **{"A-r09999-c00": 1})),             # chunk that does not exist
            label(grades={"A-r00001-c00": 2}),                          # pool chunks left ungraded
            label(answerability="unanswerable", verdict="unknown"),     # unanswerable with a grade-2 chunk
            label(grades=ALL_ZERO),                                     # answered without grade 2
            label(verdict=None),                                        # yesno without verdict
            label(qid="qd", verdict="yes"),                             # descriptive with verdict
            label(verdict="unknown"),                                   # unknown only for unanswerable
            label(answerability="unanswerable", verdict="no", grades=ALL_ZERO),
            label(grades=dict(two, **{"A-r00000-c00": 3})),             # grade out of range
            label(answerability="maybe"),
            label(rationale="  "),
        ]
        for b in bad:
            with self.subTest(b=b), self.assertRaises(LabelError):
                save_label(self.conn, "gold_v1", b)
        self.assertEqual(get_item(self.conn, "gold_v1", "qa")["labels"], {})

    def test_unanswerable_with_unknown_verdict_is_valid(self):
        save_label(self.conn, "gold_v1", label(answerability="unanswerable", verdict="unknown", grades=ALL_ZERO))

    def test_unknown_question_is_refused(self):
        with self.assertRaises(LabelError):
            save_label(self.conn, "gold_v1", label(qid="nope"))

    def test_blind_flow(self):
        final = label(qid="qd", source="final", verdict=None, annotator="human")
        with self.assertRaises(LabelError):
            save_label(self.conn, "gold_v1", final)  # blind item needs its blind label first
        save_label(self.conn, "gold_v1", Label(qid="qd", source="blind", answerability="partial", verdict=None,
                                               grades={}, rationale="", annotator="human"))
        with self.assertRaises(LabelError):
            save_label(self.conn, "gold_v1", Label(qid="qd", source="blind", answerability="answered", verdict=None,
                                                   grades={}, rationale="", annotator="human"))
        save_label(self.conn, "gold_v1", final)
        row = export_labels(self.conn, "gold_v1")[0]
        self.assertEqual(row["blind"], {"answerability": "partial", "verdict": None})
        self.assertIsNone(row["draft"])
        self.assertTrue(row["changed_from_draft"])  # no draft to agree with

    def test_list_items_status_and_no_label_content(self):
        self.assertEqual([i["status"] for i in list_items(self.conn, "gold_v1")], ["pending", "pending"])
        save_label(self.conn, "gold_v1", label())
        save_label(self.conn, "gold_v1", label(qid="qd", verdict=None))
        save_label(self.conn, "gold_v1", label(source="final"))
        items = list_items(self.conn, "gold_v1")
        self.assertEqual({i["qid"]: i["status"] for i in items}, {"qa": "done", "qd": "drafted"})
        for i in items:
            self.assertEqual(set(i), {"qid", "split", "blind", "status"})

    def test_reingest_cannot_cascade_delete_annotated_question(self):
        save_label(self.conn, "gold_v1", label())
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            load_dataset(self.conn, make_manifest("A", "B"), {"A": make_product("A"), "B": make_product("B")},
                         {"A": [question("qd", "descriptive")]})
        self.conn.rollback()
        self.assertIn("draft", get_item(self.conn, "gold_v1", "qa")["labels"])

    def test_get_item_reports_pool_chunks_missing_from_the_database(self):
        self.assertEqual(get_item(self.conn, "gold_v1", "qa")["missing_chunks"], [])
        self.conn.execute("DELETE FROM chunks WHERE chunk_id = 'A-r00002-c00'")
        self.conn.commit()
        item = get_item(self.conn, "gold_v1", "qa")
        self.assertEqual(item["missing_chunks"], ["A-r00002-c00"])
        self.assertEqual([c["chunk_id"] for c in item["pool"]], ["A-r00000-c00", "A-r00001-c00"])

    def test_draft_cannot_be_replaced_after_human_labels(self):
        save_label(self.conn, "gold_v1", label())
        save_label(self.conn, "gold_v1", label(rationale="corrected draft"))  # still only a draft: allowed
        save_label(self.conn, "gold_v1", label(source="final", annotator="human"))
        with self.assertRaises(LabelError):
            save_label(self.conn, "gold_v1", label(rationale="late draft"))
        self.assertEqual(get_item(self.conn, "gold_v1", "qa")["labels"]["draft"]["rationale"], "corrected draft")

    def test_malformed_label_values_rejected(self):
        two = dict(ALL_ZERO, **{"A-r00001-c00": 2})
        for bad in (label(grades=dict(two, **{"A-r00000-c00": True})), label(rationale=None),
                    label(grades=list(two))):
            with self.subTest(bad=bad), self.assertRaises(LabelError):
                save_label(self.conn, "gold_v1", bad)

    def test_add_items_with_unknown_qid_refused(self):
        with self.assertRaises(LabelError):
            add_items(self.conn, "gold_v2", [ItemSpec("qa", "dev", False, POOL), ItemSpec("nope", "dev", False, ())])
        self.assertEqual(list_items(self.conn, "gold_v2"), [])

    def test_add_items_twice_refused(self):
        with self.assertRaises(LabelError):
            add_items(self.conn, "gold_v1", [ItemSpec("qa", "dev", False, POOL)])


if __name__ == "__main__":
    unittest.main()
