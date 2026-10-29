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

from reviewproof.ingest.titles import load_titles, main, scan_metadata, scan_metadata_2014
from tests.support import make_manifest


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def meta_line(parent_asin: str, title: str, store: str = "Acme", compact: bool = False) -> bytes:
    record = {"main_category": "All Electronics", "title": title, "description": ["see parent_asin B0WANTED01"],
              "store": store, "parent_asin": parent_asin, "bought_together": None}
    separators = (",", ":") if compact else (", ", ": ")
    return json.dumps(record, separators=separators).encode() + b"\n"


def meta_line_2014(asin: str, title: str | None, brand: str | None = "Acme") -> bytes:
    # The 2014 files are Python literals (single quotes), not JSON.
    record = {"asin": asin, "categories": [["Home & Kitchen", "Kitchen & Dining"]],
              "related": {"also_bought": ["B0WANTED01"]}}
    if title is not None:
        record["title"] = title
    if brand is not None:
        record["brand"] = brand
    return repr(record).encode() + b"\n"


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


class ScanMetadata2014Tests(unittest.TestCase):
    def test_reads_python_literal_records(self):
        found = scan_metadata_2014([meta_line_2014("B0OTHER001", "Other"),
                                    meta_line_2014("B0WANTED01", "Men's Pepper Mill", "Atlas")], {"B0WANTED01"})
        self.assertEqual(found, {"B0WANTED01": {"title": "Men's Pepper Mill", "store": "Atlas",
                                                "main_category": "Home & Kitchen"}})

    def test_asin_in_related_is_not_a_match(self):
        self.assertEqual(scan_metadata_2014([meta_line_2014("B0OTHER001", "Other")], {"B0WANTED01"}), {})

    def test_record_without_title_is_skipped_and_brand_is_optional(self):
        found = scan_metadata_2014([meta_line_2014("B0WANTED01", None), meta_line_2014("B0WANTED02", "Mill", None)],
                                   {"B0WANTED01", "B0WANTED02"})
        self.assertEqual(found, {"B0WANTED02": {"title": "Mill", "store": None, "main_category": "Home & Kitchen"}})

    def test_html_entities_and_extra_spaces_are_normalized(self):
        found = scan_metadata_2014([meta_line_2014("B0WANTED01", "Atlas&reg;  Pepper Mill, 9&quot; ", "Meguiar&#39;s")],
                                   {"B0WANTED01"})
        self.assertEqual(found["B0WANTED01"]["title"], 'Atlas® Pepper Mill, 9"')
        self.assertEqual(found["B0WANTED01"]["store"], "Meguiar's")

    def test_code_in_a_record_is_not_run(self):
        # literal_eval refuses anything but literals; such a line is skipped, never evaluated.
        line = b"{'asin': 'B0WANTED01', 'title': __import__('os').getcwd()}\n"
        self.assertEqual(scan_metadata_2014([line], {"B0WANTED01"}), {})


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
        self.out = self.root / "manifest_v1.titles.json"

    def serve(self, name: str, lines: list[bytes]) -> None:
        (self.served / name).write_bytes(gzip.compress(b"".join(lines)))

    def run_main(self, *args: str) -> int:
        return main(["--manifest", str(self.manifest), "--base-url", self.base_url, *args])

    def test_writes_titles_and_missing_next_to_the_manifest(self):
        self.serve("meta_Electronics.jsonl.gz", [meta_line("B0WANTED01", "Koss PortaPro", "Koss"),
                                                 meta_line("B0NOPE0001", "x")])
        self.serve("meta_Automotive.jsonl.gz", [meta_line("B0WANTED02", "Lift Support")])
        code = self.run_main("--sources", "2023", "--categories", "Electronics", "Automotive")
        self.assertEqual(code, 0)
        out = json.loads(self.out.read_text())
        self.assertEqual(out["dataset_id"], "test-dataset")
        self.assertEqual(sorted(out["titles"]), ["B0WANTED01", "B0WANTED02"])
        self.assertEqual(out["titles"]["B0WANTED01"]["file"], "Electronics")
        self.assertEqual(out["titles"]["B0WANTED01"]["source"], "2023")
        self.assertEqual(out["missing"], ["B0MISSING1"])
        self.assertEqual(out["sources"]["2023"]["files"],
                         [f"{self.base_url}/meta_Electronics.jsonl.gz", f"{self.base_url}/meta_Automotive.jsonl.gz"])
        self.assertEqual(load_titles(self.out), {"B0WANTED01": "Koss PortaPro", "B0WANTED02": "Lift Support"})

    def test_later_source_fills_only_missing_titles_and_keeps_existing_ones(self):
        self.serve("meta_Electronics.jsonl.gz", [meta_line("B0WANTED01", "Koss PortaPro", "Koss")])
        self.assertEqual(self.run_main("--sources", "2023", "--categories", "Electronics"), 0)
        self.serve("meta_Electronics.json.gz", [meta_line_2014("B0WANTED01", "Old Koss Title"),
                                                meta_line_2014("B0MISSING1", "Pepper Mill", "Atlas")])
        self.assertEqual(self.run_main("--sources", "2014", "--categories", "Electronics"), 0)
        out = json.loads(self.out.read_text())
        self.assertEqual(out["titles"]["B0WANTED01"]["title"], "Koss PortaPro")
        self.assertEqual(out["titles"]["B0MISSING1"],
                         {"title": "Pepper Mill", "store": "Atlas", "main_category": "Home & Kitchen",
                          "file": "Electronics", "source": "2014"})
        self.assertEqual(out["missing"], ["B0WANTED02"])
        self.assertEqual(sorted(out["sources"]), ["2014", "2023"])
        self.assertEqual(list(out["titles"]), ["B0WANTED01", "B0MISSING1"])  # manifest order

    def test_failed_file_writes_nothing(self):
        # A partial scan would list found products as missing; refuse instead of writing it.
        self.serve("meta_Electronics.jsonl.gz", [meta_line("B0WANTED01", "Koss PortaPro")])
        code = self.run_main("--sources", "2023", "--categories", "Electronics", "Automotive")
        self.assertEqual(code, 2)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
