"""Look up the sampled products' titles in Amazon item metadata:
python -m reviewproof.ingest.titles --manifest samples/manifest_v1.json

AmazonQA has no product names. Two McAuley Lab metadata releases are read in priority order:
Amazon Reviews 2023 (Hou et al., 2024), keyed by parent_asin, which is what the older datasets call
asin; then, for products still without a title, the 2014 Amazon product data AmazonQA was built from
(McAuley et al., SIGIR 2015), keyed by asin and written as Python literals, not JSON. Each category
file is streamed and decompressed in memory, never saved; only the sampled products' title, store
(brand) and main category are written, to <manifest>.titles.json next to the manifest, which ingest
reads. When that file exists its titles are kept and only products without one are looked up;
delete it to rebuild from scratch.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import gzip
import http.client
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Iterable, Iterator, Sequence

from reviewproof.text import normalize_text

_PARENT_ASIN = re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')
_ASIN_2014 = re.compile(rb"'asin':\s*'([^']+)'")


def titles_path(manifest: Path) -> Path:
    return manifest.with_suffix(".titles.json")


def _entry(title, store, main_category) -> dict | None:
    # The 2014 metadata keeps HTML entities (&reg;, &quot;, &#39;) in titles and brands.
    title = normalize_text(title or "")
    store = normalize_text(store) or None if store else None
    return {"title": title, "store": store, "main_category": main_category} if title else None


def scan_metadata(lines: Iterable[bytes], asins: Collection[str]) -> dict[str, dict]:
    """Amazon Reviews 2023 lines → {parent_asin: {title, store, main_category}} for the wanted products
    that have a title. Only lines whose parent_asin is wanted are parsed, so a multi-GB file costs one
    regex match per line."""
    found: dict[str, dict] = {}
    for line in lines:
        start = line.rfind(b'"parent_asin"')
        match = _PARENT_ASIN.match(line, start) if start >= 0 else None
        if not match or match.group(1).decode() not in asins:
            continue
        record = json.loads(line)
        entry = _entry(record.get("title"), record.get("store"), record.get("main_category"))
        if record.get("parent_asin") in asins and entry and record["parent_asin"] not in found:
            found[record["parent_asin"]] = entry
    return found


def scan_metadata_2014(lines: Iterable[bytes], asins: Collection[str]) -> dict[str, dict]:
    """The same for the 2014 metadata, whose lines are Python dict literals. They are parsed with
    ast.literal_eval, which accepts literals only, so a line can never run code; one it refuses is skipped."""
    found: dict[str, dict] = {}
    for line in lines:
        match = _ASIN_2014.search(line)
        if not match or match.group(1).decode() not in asins:
            continue
        try:
            record = ast.literal_eval(line.decode("utf-8", errors="replace"))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        categories = record.get("categories") or [[]]
        entry = _entry(record.get("title"), record.get("brand"), categories[0][0] if categories[0] else None)
        if record.get("asin") in asins and entry and record["asin"] not in found:
            found[record["asin"]] = entry
    return found


@dataclass(frozen=True)
class Source:
    dataset: str
    homepage: str
    citation: str
    base_url: str
    file_pattern: str
    categories: tuple[str, ...]
    scan: Callable[[Iterable[bytes], Collection[str]], dict[str, dict]]


SOURCES = {
    "2023": Source(
        "Amazon Reviews 2023", "https://amazon-reviews-2023.github.io", "Hou et al., 2024, arXiv:2403.03952",
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories", "meta_{category}.jsonl.gz",
        # AmazonQA's Beauty became two categories in 2023; reading every file also covers products filed elsewhere.
        ("All_Beauty", "Beauty_and_Personal_Care", "Electronics", "Home_and_Kitchen", "Automotive"), scan_metadata),
    "2014": Source(
        "Amazon product data (2014)", "https://snap.stanford.edu/data/amazon/productGraph/",
        "McAuley, Targett, Shi, van den Hengel. Image-based recommendations on styles and substitutes. SIGIR 2015",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles", "meta_{category}.json.gz",
        ("Beauty", "Electronics", "Home_and_Kitchen", "Automotive"), scan_metadata_2014),
}


def stream_gzip_lines(url: str, *, timeout: float = 60.0) -> Iterator[bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        yield from gzip.GzipFile(fileobj=response)


def _scan_file(source: Source, category: str, url: str, asins: Collection[str], timeout: float) -> dict:
    found = source.scan(stream_gzip_lines(url, timeout=timeout), asins)
    print(f"{category}: {len(found)} found", file=sys.stderr, flush=True)
    return found


def load_titles(path: Path) -> dict[str, str]:
    """{product_id: title} from a titles file written by main()."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {pid: entry["title"] for pid, entry in data["titles"].items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Look up sampled products' titles in Amazon item metadata.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                        help="metadata releases to read, in priority order")
    parser.add_argument("--base-url", help="read every source's files from this URL instead (for tests)")
    parser.add_argument("--categories", nargs="+", help="category files to read; defaults to each source's own")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per connect or read")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids = [p["product_id"] for p in manifest["products"]]
    out = titles_path(args.manifest)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    titles: dict[str, dict] = dict(existing.get("titles", {}))
    sources: dict[str, dict] = dict(existing.get("sources", {}))

    for name in args.sources:
        source = SOURCES[name]
        wanted = {pid for pid in ids if pid not in titles}
        if not wanted:
            break
        base = (args.base_url or source.base_url).rstrip("/")
        urls = {c: f"{base}/{source.file_pattern.format(category=c)}" for c in (args.categories or source.categories)}
        failed = []
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            futures = {c: pool.submit(_scan_file, source, c, url, wanted, args.timeout) for c, url in urls.items()}
            found_by_file = {}
            for category, future in futures.items():
                try:
                    found_by_file[category] = future.result()
                except (OSError, EOFError, http.client.HTTPException) as exc:
                    failed.append(f"{urls[category]}: {exc}")
        if failed:
            # A partial scan would report products in the unread files as missing.
            print("读取元数据失败，没有写入结果:\n  " + "\n  ".join(failed), file=sys.stderr)
            return 2
        for category, found in found_by_file.items():
            for pid, entry in found.items():
                titles.setdefault(pid, {**entry, "file": category, "source": name})
        sources[name] = {"dataset": source.dataset, "homepage": source.homepage, "citation": source.citation,
                         "files": list(urls.values()), "retrieved": datetime.date.today().isoformat()}

    out.write_text(json.dumps({
        "sources": sources,
        "dataset_id": manifest["dataset_id"],
        "titles": {pid: titles[pid] for pid in ids if pid in titles},
        "missing": [pid for pid in ids if pid not in titles],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_source = {name: sum(1 for e in titles.values() if e.get("source") == name) for name in sources}
    print(f"{out}: {len(titles)}/{len(ids)} products have a title {by_source}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
