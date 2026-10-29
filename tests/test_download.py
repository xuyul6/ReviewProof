import contextlib
import functools
import hashlib
import io
import json
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from reviewproof.ingest.download import DownloadError, ensure_file, main
from tests.support import make_manifest


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.served = Path(tmp.name) / "served"
        self.local = Path(tmp.name) / "local"
        self.served.mkdir()
        self.local.mkdir()
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(self.served)))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        # Progress goes to stderr; keep test output clean.
        stderr = contextlib.redirect_stderr(io.StringIO())
        stderr.__enter__()
        self.addCleanup(stderr.__exit__, None, None, None)

    def serve(self, name: str, data: bytes) -> str:
        (self.served / name).write_bytes(data)
        return f"{self.base_url}/{name}"

    def test_downloads_and_verifies(self):
        url = self.serve("p.jsonl", b"reviews\n")
        target = self.local / "p.jsonl"
        self.assertEqual(ensure_file(target, url, sha(b"reviews\n")), "downloaded")
        self.assertEqual(target.read_bytes(), b"reviews\n")
        self.assertEqual(sorted(p.name for p in self.local.iterdir()), ["p.jsonl"])

    def test_present_file_with_matching_sha_is_not_fetched(self):
        target = self.local / "p.jsonl"
        target.write_bytes(b"reviews\n")
        # Nothing is served, so any request would fail with 404.
        self.assertEqual(ensure_file(target, f"{self.base_url}/p.jsonl", sha(b"reviews\n")), "present")

    def test_sha_mismatch_after_download_leaves_no_file(self):
        # An interrupted or corrupted download must never be left where ingest would read it.
        url = self.serve("p.jsonl", b"truncated")
        target = self.local / "p.jsonl"
        with self.assertRaises(DownloadError):
            ensure_file(target, url, sha(b"reviews\n"))
        self.assertEqual(list(self.local.iterdir()), [])

    def test_existing_file_with_wrong_sha_is_refused_and_kept(self):
        # A different local copy is the user's file: report it, never overwrite it.
        url = self.serve("p.jsonl", b"reviews\n")
        target = self.local / "p.jsonl"
        target.write_bytes(b"my own edit\n")
        with self.assertRaises(DownloadError) as ctx:
            ensure_file(target, url, sha(b"reviews\n"))
        self.assertIn(str(target), str(ctx.exception))
        self.assertEqual(target.read_bytes(), b"my own edit\n")

    def test_http_error_names_the_url(self):
        url = f"{self.base_url}/missing.jsonl"
        with self.assertRaises(DownloadError) as ctx:
            ensure_file(self.local / "missing.jsonl", url, sha(b""))
        self.assertIn(url, str(ctx.exception))
        self.assertEqual(list(self.local.iterdir()), [])

    def test_main_fetches_both_manifest_files(self):
        self.serve("p.jsonl", b"products\n")
        self.serve("q.jsonl", b"questions\n")
        manifest = make_manifest("A")
        manifest["source"] = {"products_file": str(self.local / "data" / "p.jsonl"), "products_sha256": sha(b"products\n"),
                              "questions_file": str(self.local / "data" / "q.jsonl"), "questions_sha256": sha(b"questions\n")}
        path = self.local / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(main(["--manifest", str(path), "--base-url", self.base_url]), 0)
        self.assertEqual((self.local / "data" / "q.jsonl").read_bytes(), b"questions\n")

        manifest["source"]["products_sha256"] = sha(b"other\n")
        path.write_text(json.dumps(manifest))
        self.assertEqual(main(["--manifest", str(path), "--base-url", self.base_url]), 2)


if __name__ == "__main__":
    unittest.main()
