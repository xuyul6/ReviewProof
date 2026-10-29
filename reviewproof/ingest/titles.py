"""Look up the sampled products' titles in the Amazon Reviews 2023 item metadata:
python -m reviewproof.ingest.titles --manifest samples/manifest_v1.json

AmazonQA has no product names. Amazon Reviews 2023 (McAuley Lab, Hou et al., 2024, arXiv:2403.03952)
keys its metadata by parent_asin, which is what the older datasets AmazonQA comes from call asin.
Each category file (6.3 GB compressed for the five read here) is streamed and decompressed in memory,
never saved; only the sampled products' title, store and main_category are written, to
<manifest>.titles.json next to the manifest, which ingest reads.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import http.client
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Collection, Iterable, Iterator, Sequence

HOMEPAGE = "https://amazon-reviews-2023.github.io"
CITATION = "Hou et al., 2024, arXiv:2403.03952"
META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
# AmazonQA's Beauty became two categories in 2023; scanning every file also covers products filed elsewhere.
META_CATEGORIES = ("All_Beauty", "Beauty_and_Personal_Care", "Electronics", "Home_and_Kitchen", "Automotive")

_PARENT_ASIN = re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')


def titles_path(manifest: Path) -> Path:
    return manifest.with_suffix(".titles.json")


def scan_metadata(lines: Iterable[bytes], asins: Collection[str]) -> dict[str, dict]:
    """Return {parent_asin: {title, store, main_category}} for the wanted products that have a title.
    Only lines whose parent_asin is wanted are parsed, so a multi-GB file costs one regex search per line."""
    found: dict[str, dict] = {}
    for line in lines:
        start = line.rfind(b'"parent_asin"')
        match = _PARENT_ASIN.match(line, start) if start >= 0 else None
        if not match or match.group(1).decode() not in asins:
            continue
        record = json.loads(line)
        title = (record.get("title") or "").strip()
        if record.get("parent_asin") in asins and title and record["parent_asin"] not in found:
            found[record["parent_asin"]] = {"title": title, "store": record.get("store"),
                                            "main_category": record.get("main_category")}
    return found


def stream_gzip_lines(url: str, *, timeout: float = 60.0) -> Iterator[bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        yield from gzip.GzipFile(fileobj=response)


def _scan_file(category: str, url: str, asins: Collection[str], timeout: float) -> tuple[str, dict]:
    found = scan_metadata(stream_gzip_lines(url, timeout=timeout), asins)
    print(f"{category}: {len(found)} found", file=sys.stderr, flush=True)
    return category, found


def load_titles(path: Path) -> dict[str, str]:
    """{product_id: title} from a titles file written by main()."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {pid: entry["title"] for pid, entry in data["titles"].items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Look up sampled products' titles in Amazon Reviews 2023 metadata.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default=META_BASE_URL)
    parser.add_argument("--categories", nargs="+", default=list(META_CATEGORIES))
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per connect or read")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids = [p["product_id"] for p in manifest["products"]]
    urls = {c: f"{args.base_url.rstrip('/')}/meta_{c}.jsonl.gz" for c in args.categories}
    titles: dict[str, dict] = {}
    failed = []
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(_scan_file, c, url, set(ids), args.timeout) for c, url in urls.items()]
        for future in futures:
            try:
                category, found = future.result()
            except (OSError, EOFError, http.client.HTTPException) as exc:
                failed.append(str(exc))
                continue
            for pid, entry in found.items():
                titles.setdefault(pid, {**entry, "file": category})
    if failed:
        # A partial scan would report products in the unread files as missing.
        print("读取元数据失败，没有写入结果:\n  " + "\n  ".join(failed), file=sys.stderr)
        return 2

    out = titles_path(args.manifest)
    out.write_text(json.dumps({
        "source": {"dataset": "Amazon Reviews 2023", "homepage": HOMEPAGE, "citation": CITATION,
                   "files": list(urls.values()), "retrieved": datetime.date.today().isoformat()},
        "dataset_id": manifest["dataset_id"],
        "titles": {pid: titles[pid] for pid in ids if pid in titles},
        "missing": [pid for pid in ids if pid not in titles],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {len(titles)}/{len(ids)} products have a title", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
