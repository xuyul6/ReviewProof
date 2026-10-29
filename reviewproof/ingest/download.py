"""Download the two AmazonQA source files a manifest names, checked against its SHA-256 values:
python -m reviewproof.ingest.download --manifest samples/manifest_v1.json

The data is fetched from AmazonQA's own bucket and never redistributed from this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

from reviewproof.ingest.amazonqa import file_sha256

AMAZONQA_BASE_URL = "https://amazon-qa.s3-us-west-2.amazonaws.com"
_BLOCK = 1 << 20


class DownloadError(Exception):
    pass


def ensure_file(path: Path, url: str, sha256: str, *, timeout: float = 60.0) -> str:
    """Return "present" if path already holds the expected bytes, else download it and return
    "downloaded". The download goes to a .part file renamed only once its SHA-256 matches, so an
    interrupted or corrupted download never sits where ingest reads; a local file with other
    contents is the user's and is refused, never overwritten. `timeout` bounds each connect and
    read (not the whole download), so a stalled connection fails instead of hanging."""
    if path.exists():
        actual = file_sha256(path)
        if actual == sha256:
            return "present"
        raise DownloadError(f"本地文件和清单不一致，没有覆盖它: {path}\n"
                            f"  expected sha256 {sha256}\n  actual   sha256 {actual}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done, next_report = 0, 0.1
            for block in iter(lambda: response.read(_BLOCK), b""):
                out.write(block)
                digest.update(block)
                done += len(block)
                if total and done / total >= next_report:
                    print(f"{path.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", file=sys.stderr)
                    next_report += 0.1
    except BaseException as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, (OSError, http.client.HTTPException)):
            raise DownloadError(f"下载失败: {url}\n  {exc}") from exc
        raise
    actual = digest.hexdigest()
    if actual != sha256:
        part.unlink()
        raise DownloadError(f"下载的文件和清单不一致: {url}\n  expected sha256 {sha256}\n  actual   sha256 {actual}")
    part.replace(path)
    return "downloaded"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the AmazonQA source files named in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default=AMAZONQA_BASE_URL)
    args = parser.parse_args(argv)
    source = json.loads(args.manifest.read_text(encoding="utf-8"))["source"]
    try:
        for key in ("products", "questions"):
            path = Path(source[f"{key}_file"])
            result = ensure_file(path, f"{args.base_url.rstrip('/')}/{path.name}", source[f"{key}_sha256"])
            print(f"{path}: {'已存在' if result == 'present' else '已下载'}，SHA-256 与清单一致", file=sys.stderr)
    except DownloadError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
