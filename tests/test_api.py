import dataclasses
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from reviewproof.api.app import create_app
from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.loader import load_dataset
from reviewproof.runs import new_run_id, save_run
from reviewproof.workflow.graph import QAResult
from tests.support import DatabaseTestCase, make_manifest, make_product, make_test_services

SECRET_Q = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno",
                    is_answerable=True, answers=({"answerText": "SECRET-ANSWER"},),
                    review_snippets=("SECRET-SNIPPET",))
NO_FRONTEND = Path("/nonexistent-frontend-dist")


def manifest_with_splits():
    manifest = make_manifest("A", "B")
    manifest["products"][1].update(category="Beauty", split="test")
    return manifest


class ApiTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        long_texts = ["word " * 2000 + str(i) for i in range(4)]  # ~10k tokens → hybrid
        load_dataset(self.conn, manifest_with_splits(),
                     {"A": make_product("A"),
                      "B": dataclasses.replace(make_product("B", long_texts), category="Beauty")}, {"A": [SECRET_Q]})
        self.client = TestClient(create_app(make_test_services(), static_dir=NO_FRONTEND))

    def test_healthz(self):
        body = self.client.get("/healthz").json()
        self.assertEqual((body["database"], body["llm_model"]), (True, "fake-llm"))
        self.assertIn("anthropic_key_set", body)

    def test_products_list_and_filters(self):
        body = self.client.get("/v1/products").json()
        self.assertIn("test-dataset", body["source"])
        products = {p["product_id"]: p for p in body["products"]}
        self.assertEqual(set(products), {"A", "B"})
        self.assertEqual((products["A"]["strategy"], products["B"]["strategy"]), ("full_context", "hybrid"))
        self.assertEqual((products["A"]["review_count"], products["A"]["title"]), (3, None))
        self.assertEqual([p["product_id"] for p in self.client.get("/v1/products?category=Beauty").json()["products"]],
                         ["B"])
        self.assertEqual([p["product_id"] for p in self.client.get("/v1/products?split=dev").json()["products"]],
                         ["A"])

    def test_questions_only_text(self):
        response = self.client.get("/v1/products/A/questions")
        self.assertEqual(response.json(), {"product_id": "A",
                                           "questions": [{"qid": "q1", "question_text": "Is it loud?"}]})
        for secret in ("SECRET-ANSWER", "SECRET-SNIPPET", "is_answerable"):
            self.assertNotIn(secret, response.text)

    def test_questions_unknown_product_404(self):
        self.assertEqual(self.client.get("/v1/products/ZZZ/questions").status_code, 404)

    def test_evidence_returns_review_and_chunks(self):
        body = self.client.get("/v1/evidence/A-r00001").json()
        self.assertEqual((body["product_id"], body["text"], body["helpful_yes"], body["helpful_total"]),
                         ("A", "Too loud for me.", 1, 2))
        self.assertEqual(body["chunks"], [{"chunk_id": "A-r00001-c00", "start_offset": 0, "end_offset": 16}])

    def test_evidence_missing_404(self):
        response = self.client.get("/v1/evidence/A-r09999")
        self.assertEqual((response.status_code, response.json()["detail"]),
                         (404, "review not found in the current data"))

    def test_run_round_trip_and_404(self):
        result = QAResult(product_id="A", question="Is it loud?", route="review_question", status="unanswerable",
                          service_status="ok", verdict="unknown", claims=(), unknowns=(), citations=(),
                          answer_text="The reviews don't confirm this.", search_queries=("loud",),
                          retrieval_modes=("full_context",), rewrites_used=0, trace=(), llm_calls=2,
                          input_tokens=20, output_tokens=10, latency_ms=5.0)
        run_id = new_run_id()
        save_run(self.conn, run_id, result, "fake-llm")
        body = self.client.get(f"/v1/runs/{run_id}").json()
        self.assertEqual((body["run_id"], body["answer_text"]), (run_id, "The reviews don't confirm this."))
        self.assertEqual(self.client.get("/v1/runs/nope").status_code, 404)

    def test_serves_built_frontend(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "index.html").write_text("<html>ReviewProof UI</html>", encoding="utf-8")
            client = TestClient(create_app(make_test_services(), static_dir=Path(tmp)))
            self.assertIn("ReviewProof UI", client.get("/").text)
            self.assertEqual(client.get("/healthz").status_code, 200)

    def test_no_frontend_build_is_404(self):
        self.assertEqual(self.client.get("/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
