import asyncio
import json
import unittest

from reviewproof.ingest.amazonqa import Question
from reviewproof.ingest.embed import embed_pending_chunks
from reviewproof.ingest.loader import load_dataset
from reviewproof.mcp_server import build_server, tools
from tests.support import DatabaseTestCase, FakeEmbedder, make_manifest, make_product, make_test_services

SECRET_Q = Question(qid="q1", product_id="A", question_text="Is it loud?", question_type="yesno",
                    is_answerable=True, answers=({"answerText": "SECRET-ANSWER"},),
                    review_snippets=("SECRET-SNIPPET",))


class McpServerTests(DatabaseTestCase):
    def setUp(self):
        super().setUp()
        load_dataset(self.conn, make_manifest("A", "B"),
                     {"A": make_product("A"), "B": make_product("B", ["Too loud for me.", "Cheap plastic."])},
                     {"A": [SECRET_Q]})
        embed_pending_chunks(self.conn, FakeEmbedder(), batch_size=10)
        self.services = make_test_services()
        self.tools = tools(self.services)

    def test_tools_registered(self):
        names = {t.name for t in asyncio.run(build_server(self.services).list_tools())}
        self.assertEqual(names, {"list_products", "search_reviews", "get_review"})

    def test_list_products(self):
        products = self.tools["list_products"]()
        self.assertEqual({p["product_id"] for p in products}, {"A", "B"})
        self.assertEqual(self.tools["list_products"](category="Nope"), [])

    def test_search_reviews_scoped_to_product(self):
        body = self.tools["search_reviews"]("A", "Too loud for me.")
        self.assertEqual((body["strategy"], body["service_status"]), ("full_context", "ok"))
        self.assertTrue(all(r["chunk_id"].startswith("A-") for r in body["results"]))
        self.services.retrieval = self.services.retrieval.__class__(
            **{**self.services.retrieval.__dict__,
               "params": self.services.retrieval.params.__class__(full_context_max_tokens=0)})
        hybrid = tools(self.services)["search_reviews"]("A", "Too loud for me.")
        self.assertEqual((hybrid["strategy"], hybrid["results"][0]["chunk_id"]), ("hybrid", "A-r00001-c00"))

    def test_unknown_product_and_review_return_error(self):
        self.assertEqual(self.tools["search_reviews"]("ZZZ", "loud"), {"error": "product not found"})
        self.assertEqual(self.tools["get_review"]("A-r09999"), {"error": "review not found in the current data"})
        self.assertEqual(self.tools["get_review"]("A-r00001")["text"], "Too loud for me.")

    def test_no_community_data_in_any_tool_output(self):
        outputs = [self.tools["list_products"](), self.tools["search_reviews"]("A", "loud"),
                   self.tools["get_review"]("A-r00001")]
        self.assertNotIn("SECRET", json.dumps(outputs))


if __name__ == "__main__":
    unittest.main()
