import dataclasses
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from reviewproof.annotate.store import ItemSpec, Label, add_items, save_label
from reviewproof.api.app import create_app
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from tests.support import DatabaseTestCase, make_manifest, make_product, make_test_services

NO_FRONTEND = Path("/nonexistent-frontend-dist")
POOL = ("A-r00000-c00", "A-r00001-c00")
DRAFT_GRADES = {"A-r00000-c00": 0, "A-r00001-c00": 2}


def question(qid, qtype):
    return Question(qid=qid, product_id="A", question_text="Is it loud?", question_type=qtype, is_answerable=True,
                    answers=({"answerText": "COMMUNITY-ANSWER"},), review_snippets=("SECRET-SNIPPET",))


class AnnotationApiTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A"), {"A": make_product("A")},
                     {"A": [question("qa", "yesno"), question("qd", "descriptive")]})
        add_items(self.conn, "gold_v1", [ItemSpec("qa", "dev", False, POOL), ItemSpec("qd", "dev", True, POOL)])
        for qid, verdict in (("qa", "yes"), ("qd", None)):
            save_label(self.conn, "gold_v1", Label(qid, "draft", "answered", verdict, DRAFT_GRADES,
                                                   "SECRET-DRAFT-RATIONALE", "agent:test"))
        services = make_test_services()
        services.settings = dataclasses.replace(services.settings, annotation_enabled=True, annotator="lux")
        self.client = TestClient(create_app(services, static_dir=NO_FRONTEND))
        self.base = "/v1/annotation/gold_v1"

    def test_routes_absent_when_disabled(self):
        client = TestClient(create_app(make_test_services(), static_dir=NO_FRONTEND))
        self.assertEqual(client.get(f"{self.base}/items").status_code, 404)

    def test_list_items(self):
        body = self.client.get(f"{self.base}/items").json()
        self.assertEqual([(i["qid"], i["status"], i["blind"]) for i in body["items"]],
                         [("qa", "drafted", False), ("qd", "drafted", True)])
        self.assertNotIn("SECRET-DRAFT-RATIONALE", self.client.get(f"{self.base}/items").text)

    def test_open_item_shows_draft_pool_and_community_answers(self):
        body = self.client.get(f"{self.base}/items/qa").json()
        self.assertFalse(body["draft_hidden"])
        self.assertEqual(body["labels"]["draft"]["grades"], DRAFT_GRADES)
        self.assertEqual([c["chunk_id"] for c in body["pool"]], list(POOL))
        self.assertEqual(body["community_answers"], ["COMMUNITY-ANSWER"])

    def test_blind_item_hides_draft_until_blind_label(self):
        response = self.client.get(f"{self.base}/items/qd")
        self.assertTrue(response.json()["draft_hidden"])
        self.assertNotIn("draft", response.json()["labels"])
        self.assertNotIn("SECRET-DRAFT-RATIONALE", response.text)
        posted = self.client.post(f"{self.base}/items/qd/blind", json={"answerability": "partial", "verdict": None})
        self.assertEqual(posted.status_code, 200)
        self.assertFalse(posted.json()["draft_hidden"])
        self.assertEqual(posted.json()["labels"]["draft"]["rationale"], "SECRET-DRAFT-RATIONALE")
        self.assertEqual(posted.json()["labels"]["blind"]["annotator"], "lux")
        again = self.client.post(f"{self.base}/items/qd/blind", json={"answerability": "answered", "verdict": None})
        self.assertEqual(again.status_code, 409)

    def test_blind_on_regular_item_is_409(self):
        response = self.client.post(f"{self.base}/items/qa/blind", json={"answerability": "answered", "verdict": "yes"})
        self.assertEqual(response.status_code, 409)

    def test_final_saves_and_marks_done(self):
        body = {"answerability": "answered", "verdict": "yes", "grades": DRAFT_GRADES, "rationale": "r1 says so"}
        response = self.client.put(f"{self.base}/items/qa/final", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labels"]["final"]["annotator"], "lux")
        self.assertEqual(self.client.get(f"{self.base}/items").json()["items"][0]["status"], "done")

    def test_final_validation_error_is_422_with_reason(self):
        body = {"answerability": "unanswerable", "verdict": "unknown", "grades": DRAFT_GRADES, "rationale": "x"}
        response = self.client.put(f"{self.base}/items/qa/final", json=body)
        self.assertEqual(response.status_code, 422)
        self.assertIn("grade-2", response.json()["detail"])

    def test_final_before_blind_is_409(self):
        body = {"answerability": "answered", "verdict": None, "grades": DRAFT_GRADES, "rationale": "x"}
        self.assertEqual(self.client.put(f"{self.base}/items/qd/final", json=body).status_code, 409)

    def test_unknown_item_is_404(self):
        self.assertEqual(self.client.get(f"{self.base}/items/nope").status_code, 404)

    def test_item_never_contains_silver_or_snippets(self):
        for qid in ("qa", "qd"):
            text = self.client.get(f"{self.base}/items/{qid}").text
            for secret in ("SECRET-SNIPPET", "is_answerable", "silver"):
                self.assertNotIn(secret, text)

    def test_item_includes_text_of_graded_out_of_pool_chunks(self):
        save_label(self.conn, "gold_v1", Label("qa", "draft", "answered", "yes", dict(DRAFT_GRADES, **{"A-r00002-c00": 1}),
                                               "r", "agent:test"))
        body = self.client.get(f"{self.base}/items/qa").json()
        self.assertEqual(body["extra_chunks"], [{"chunk_id": "A-r00002-c00", "review_id": "A-r00002",
                                                 "text": "Fits the 4th gen iPad."}])
        self.assertEqual(self.client.get(f"{self.base}/items/qd").json()["extra_chunks"], [])

    def test_search_returns_out_of_pool_chunks(self):
        body = self.client.get(f"{self.base}/items/qa/search", params={"q": "ipad"}).json()
        self.assertEqual([c["chunk_id"] for c in body["chunks"]], ["A-r00002-c00"])


if __name__ == "__main__":
    unittest.main()
