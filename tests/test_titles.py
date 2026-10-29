import contextlib
import functools
import gzip
import io
import json
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from reviewproof.ingest.titles import load_titles, main, scan_metadata
from tests.support import make_manifest


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def meta_line(parent_asin: str, title: str, store: str = "Acme", compact: bool = False) -> bytes:
    record = {"main_category": "All Electronics", "title": title, "description": ["see parent_asin B0WANTED01"],
              "store": store, "parent_asin": parent_asin, "bought_together": None}
    separators = (",", ":") if compact else (", ", ": ")
    return json.dumps(record, separators=separators).encode() + b"\n"


class ScanMetadataTests(unittest.TestCase):
    def test_keeps_only_wanted_parent_asins(self):
        lines = [meta_line("B0OTHER001", "Other"), meta_line("B0WANTED01", "Koss PortaPro Headphones", "Koss")]
        found = scan_metadata(lines, {"B0WANTED01"})
        self.assertEqual(found, {"B0WANTED01": {"title": "Koss PortaPro Headphones", "store": "Koss",
                                                "main_category": "All Electronics"}})

    def test_asin_mentioned_in_another_field_is_not_a_match(self):
        # The description of B0OTHER001 mentions B0WANTED01; only parent_asin counts.
        self.assertEqual(scan_metadata([meta_line("B0OTHER001", "Other")], {"B0WANTED01"}), {})

    def test_compact_json_is_matched(self):
        found = scan_metadata([meta_line("B0WANTED01", "Pepper Mill", compact=True)], {"B0WANTED01"})
        self.assertEqual(found["B0WANTED01"]["title"], "Pepper Mill")

    def test_blank_title_is_not_kept(self):
        self.assertEqual(scan_metadata([meta_line("B0WANTED01", "  ")], {"B0WANTED01"}), {})


class TitlesMainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.served = self.root / "served"
        self.served.mkdir()
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(self.served)))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        stderr = contextlib.redirect_stderr(io.StringIO())
        stderr.__enter__()
        self.addCleanup(stderr.__exit__, None, None, None)
        self.manifest = self.root / "manifest_v1.json"
        self.manifest.write_text(json.dumps(make_manifest("B0WANTED01", "B0WANTED02", "B0MISSING1")))

    def serve(self, category: str, lines: list[bytes]) -> None:
        (self.served / f"meta_{category}.jsonl.gz").write_bytes(gzip.compress(b"".join(lines)))

    def test_writes_titles_and_missing_next_to_the_manifest(self):
        self.serve("Electronics", [meta_line("B0WANTED01", "Koss PortaPro", "Koss"), meta_line("B0NOPE0001", "x")])
        self.serve("Automotive", [meta_line("B0WANTED02", "Lift Support")])
        code = main(["--manifest", str(self.manifest), "--base-url", self.base_url,
                     "--categories", "Electronics", "Automotive"])
        self.assertEqual(code, 0)
        out = json.loads((self.root / "manifest_v1.titles.json").read_text())
        self.assertEqual(out["dataset_id"], "test-dataset")
        self.assertEqual(sorted(out["titles"]), ["B0WANTED01", "B0WANTED02"])
        self.assertEqual(out["titles"]["B0WANTED01"]["file"], "Electronics")
        self.assertEqual(out["missing"], ["B0MISSING1"])
        self.assertEqual(out["source"]["files"],
                         [f"{self.base_url}/meta_Electronics.jsonl.gz", f"{self.base_url}/meta_Automotive.jsonl.gz"])
        self.assertEqual(load_titles(self.root / "manifest_v1.titles.json"),
                         {"B0WANTED01": "Koss PortaPro", "B0WANTED02": "Lift Support"})

    def test_failed_file_writes_nothing(self):
        # A partial scan would list found products as missing; refuse instead of writing it.
        self.serve("Electronics", [meta_line("B0WANTED01", "Koss PortaPro")])
        code = main(["--manifest", str(self.manifest), "--base-url", self.base_url,
                     "--categories", "Electronics", "Automotive"])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "manifest_v1.titles.json").exists())


if __name__ == "__main__":
    unittest.main()
