import json
import threading
import unittest
from unittest.mock import patch

import psycopg
from fastapi.testclient import TestClient

from reviewproof.api import qa as qa_module
from reviewproof.api.app import create_app
from reviewproof.api.qa import qa_events
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.llm import LLMError
from reviewproof.workflow.schemas import Claim, EvidenceAssessment, GeneralNote, GeneratedAnswer, QueryPlan
from reviewproof.workflow.state import TraceEntry
from tests.support import DatabaseTestCase, FakeEmbedder, FakeLLM, make_manifest, make_product, make_test_services

NO_FRONTEND = None


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        name = next(l[len("event: "):] for l in lines if l.startswith("event: "))
        data = "\n".join(l[len("data: "):] for l in lines if l.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def plan_for(prompt):
    return QueryPlan(route="review_question", out_of_scope_reason="", is_yes_no=True, search_query="loud")


def assess_for(prompt):
    cited = "A-r00001-c00" if "[A-" in prompt else "B-r00000-c00"
    return EvidenceAssessment(status="answered", supporting_chunk_ids=[cited], missing=[], rewrite_query="")


def generate_for(prompt):
    cited = "A-r00001-c00" if "[A-" in prompt else "B-r00000-c00"
    return GeneratedAnswer(claims=[Claim(text="A buyer says it is too loud.", chunk_ids=[cited])], unknowns=[],
                           verdict="yes")


def script(runs=1, **overrides):
    s = {"QueryPlan": [plan_for] * runs, "EvidenceAssessment": [assess_for] * runs,
         "GeneratedAnswer": [generate_for] * runs}
    s.update(overrides)
    return s


class QaEndpointTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])}, {})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)

    def client_for(self, llm_script):
        self.services = make_test_services(FakeLLM(llm_script))
        return TestClient(create_app(self.services, static_dir=NO_FRONTEND))

    def post_qa(self, client, product_id, question):
        response = client.post("/v1/qa", json={"product_id": product_id, "question": question})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        return parse_sse(response.text)

    def count_runs(self, run_id):
        row = self.conn.execute("SELECT count(*) FROM qa_runs WHERE run_id = %s", (run_id,)).fetchone()
        self.conn.rollback()
        return row[0]

    def test_event_order_and_answer(self):
        client = self.client_for(script())
        events = self.post_qa(client, "A", "Is it loud?")
        names = [e for e, _ in events]
        self.assertEqual(names[0], "run_started")
        self.assertEqual(names[1:6], ["progress"] * 5)
        self.assertEqual([d["node"] for e, d in events if e == "progress"],
                         ["plan", "retrieve", "assess", "generate", "verify"])
        self.assertIn("answer_chunk", names)
        self.assertEqual(names[-2:], ["citations", "done"])
        text = "".join(d["text"] for e, d in events if e == "answer_chunk")
        self.assertIn("[1]", text)
        done = events[-1][1]
        self.assertEqual((done["status"], done["service_status"], done["persisted"]), ("answered", "ok", True))
        self.assertEqual(events[-2][1]["citations"][0]["chunk_id"], "A-r00001-c00")
        run = client.get(f"/v1/runs/{done['run_id']}").json()
        self.assertEqual(run["answer_text"], text)

    def test_unanswerable_run_streams_the_note(self):
        unanswerable = EvidenceAssessment(status="unanswerable", supporting_chunk_ids=[], missing=[], rewrite_query="")
        client = self.client_for(script(EvidenceAssessment=[unanswerable],
                                        GeneralNote=[GeneralNote(kind="general", text="Usually adjustable.")]))
        events = self.post_qa(client, "A", "Is it loud?")
        note = {"kind": "general",
                "heading": "Not from the reviews: what is generally true of this kind of product, for reference only.",
                "text": "Usually adjustable."}
        done = events[-1][1]
        self.assertEqual((done["status"], done["note"]), ("unanswerable", note))
        self.assertNotIn("Usually adjustable.", "".join(d["text"] for e, d in events if e == "answer_chunk"))
        self.assertEqual([d["node"] for e, d in events if e == "progress"][-1], "advise")
        self.assertEqual(client.get(f"/v1/runs/{done['run_id']}").json()["note"], note)

    def test_answered_run_has_no_note(self):
        events = self.post_qa(self.client_for(script()), "A", "Is it loud?")
        self.assertIsNone(events[-1][1]["note"])

    def test_long_lines_are_chunked_and_reassemble(self):
        long_claim = GeneratedAnswer(claims=[Claim(text="x" * 300, chunk_ids=["A-r00001-c00"])], unknowns=[],
                                     verdict="yes")
        client = self.client_for(script(GeneratedAnswer=[long_claim]))
        events = self.post_qa(client, "A", "Is it loud?")
        chunks = [d["text"] for e, d in events if e == "answer_chunk"]
        self.assertTrue(all(len(c) <= 121 for c in chunks))
        self.assertEqual("".join(chunks), "Verdict: Yes\n" + "x" * 300 + " [1]")

    def test_service_failure_is_200_and_done(self):
        client = self.client_for(script(EvidenceAssessment=[LLMError("down")]))
        events = self.post_qa(client, "A", "Is it loud?")
        done = events[-1][1]
        self.assertEqual((events[-1][0], done["service_status"], done["status"]), ("done", "failed", None))
        self.assertIn("service error", "".join(d["text"] for e, d in events if e == "answer_chunk"))
        self.assertEqual(self.count_runs(done["run_id"]), 1)

    def test_unknown_product_404_before_stream(self):
        response = self.client_for(script()).post("/v1/qa", json={"product_id": "ZZZ", "question": "Is it loud?"})
        self.assertEqual(response.status_code, 404)

    def test_blank_question_422(self):
        response = self.client_for(script()).post("/v1/qa", json={"product_id": "A", "question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_persist_failure_still_answers(self):
        client = self.client_for(script())
        with patch.object(qa_module, "save_run", side_effect=psycopg.OperationalError("disk full")):
            events = self.post_qa(client, "A", "Is it loud?")
        done = events[-1][1]
        self.assertEqual((done["status"], done["persisted"]), ("answered", False))
        self.assertIn("[1]", "".join(d["text"] for e, d in events if e == "answer_chunk"))

    def test_unexpected_exception_emits_error_event(self):
        def broken(*args, **kwargs):
            yield TraceEntry("plan", "ok", "", 1.0)
            raise RuntimeError("graph exploded")
        client = self.client_for(script())
        with patch.object(qa_module, "stream_answer", side_effect=broken):
            events = self.post_qa(client, "A", "Is it loud?")
        self.assertEqual([e for e, _ in events], ["run_started", "progress", "error"])
        self.assertIn("RuntimeError: graph exploded", events[-1][1]["message"])
        self.assertEqual(self.count_runs(events[-1][1]["run_id"]), 0)

    def test_client_disconnect_still_saves_run(self):
        services = make_test_services(FakeLLM(script()))
        gen = qa_events(services, "A", "Is it loud?")
        name, started = next(gen)
        self.assertEqual(name, "run_started")
        self.assertEqual(next(gen)[0], "progress")  # the worker thread now holds the lock
        gen.close()
        with services.lock:  # wait for the worker to finish and save
            pass
        self.assertEqual(self.count_runs(started["run_id"]), 1)

    def test_concurrent_requests_do_not_mix(self):
        client = self.client_for(script(runs=2))
        results = {}

        def ask(product_id):
            results[product_id] = self.post_qa(client, product_id, "Is it loud?")

        threads = [threading.Thread(target=ask, args=(pid,)) for pid in ("A", "B")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        run_ids = {results[pid][-1][1]["run_id"] for pid in ("A", "B")}
        self.assertEqual(len(run_ids), 2)
        for pid in ("A", "B"):
            citations = next(d for e, d in results[pid] if e == "citations")["citations"]
            self.assertTrue(citations)
            self.assertTrue(all(c["chunk_id"].startswith(pid + "-") for c in citations))


if __name__ == "__main__":
    unittest.main()
