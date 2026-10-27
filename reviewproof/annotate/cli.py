"""python -m reviewproof.annotate {pick,packet,search,import-drafts,status,export} --set LABEL_SET

pick builds the label set (questions + candidate pools) once. packet and search are the only views
the drafting agent gets: question text and review chunks, never silver labels, snippets or
community answers. import-drafts stores agent drafts after validation; export writes the final
human-reviewed labels to annotations/<set>/labels.jsonl for the evaluation command.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from reviewproof.annotate.pool import POOL_DEPTH, build_pool, pick_items, search_product
from reviewproof.annotate.store import (GUIDELINE_VERSION, ItemSpec, Label, LabelError, add_items, export_labels,
                                        get_item, list_items, save_label)
from reviewproof.config import Settings, load_settings
from reviewproof.db import apply_schema, connect
from reviewproof.embedding import Embedder
from reviewproof.eval.cli import REPO_ROOT, _git_commit, _write_jsonl
from reviewproof.eval.dataset import load_questions
from reviewproof.eval.metrics import SEED
from reviewproof.retrieval import RetrievalParams
from reviewproof.retrieval.bm25 import BM25Cache
from reviewproof.retrieval.cli import default_embedder, default_reranker
from reviewproof.retrieval.rerank import Reranker
from reviewproof.workflow.state import RetrievalDeps

DRAFT_FORMAT = f"""## 输出格式

每题输出一行 JSON：{{"qid": "...", "answerability": "answered|partial|unanswerable", "verdict": "yes|no|mixed|unknown|null", "grades": {{"<chunk_id>": 0|1|2, ...}}, "rationale": "...", "annotator": "agent:<model>"}}
候选池里的每个块都要给等级；搜索找到的池外块可以加进 grades。规则见 docs/annotation_guidelines_cn.md（{GUIDELINE_VERSION}）。
"""


def _pick(conn, args, settings: Settings, embedder_factory, reranker_factory) -> int:
    if list_items(conn, args.set):
        print(f"label set {args.set!r} already exists", file=sys.stderr)
        return 2
    by_split = {s: load_questions(conn, s, settings.full_context_max_tokens) for s in ("dev", "test")}
    picked = pick_items(by_split, {"dev": args.dev, "test": args.test}, blind_fraction=args.blind_fraction,
                        seed=args.seed)
    embedder = (embedder_factory or default_embedder)(settings)
    params = RetrievalParams(full_context_max_tokens=settings.full_context_max_tokens)
    deps = RetrievalDeps(embedder=embedder, embedding_model=embedder.model_name,
                         reranker=(reranker_factory or default_reranker)(settings), bm25_cache=BM25Cache(),
                         params=params)
    dataset_of = dict(conn.execute("SELECT product_id, dataset_id FROM products").fetchall())
    conn.rollback()
    specs, rows = [], []
    for n, (q, blind) in enumerate(picked, start=1):
        pool = build_pool(conn, q, deps)
        specs.append(ItemSpec(q.qid, q.split, blind, pool))
        rows.append({"qid": q.qid, "split": q.split, "blind": blind, "strategy": q.strategy, "pool": list(pool),
                     "dataset_id": dataset_of[q.product_id], "pool_depth": POOL_DEPTH,
                     "retrieval_params": dataclasses.asdict(params), "git_commit": _git_commit()})
        print(f"  {n}/{len(picked)} {q.qid} pool={len(pool)}", file=sys.stderr, flush=True)
    add_items(conn, args.set, specs)
    out_dir = args.out_root / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "items.jsonl", sorted(rows, key=lambda r: r["qid"]))
    print(f"{args.set}: {len(specs)} items, {sum(s.blind for s in specs)} blind")
    return 0


def _packet(conn, args) -> int:
    parts = []
    for qid in (q.strip() for q in args.qids.split(",") if q.strip()):
        item = get_item(conn, args.set, qid)
        if item is None:
            print(f"{qid} is not in label set {args.set!r}", file=sys.stderr)
            return 2
        parts += [f"## {qid}", "", f"- 品类：{item['category']}", f"- 题型：{item['question_type']}",
                  f"- 问题：{item['question_text']}", "",
                  f"候选池（{len(item['pool'])} 块，按编号排列，顺序没有含义）：", ""]
        parts += [f"[{c['chunk_id']}] {c['text']}" for c in item["pool"]] + [""]
    print("\n".join(parts + [DRAFT_FORMAT]))
    return 0


def _search(conn, args) -> int:
    item = get_item(conn, args.set, args.qid)
    if item is None:
        print(f"{args.qid} is not in label set {args.set!r}", file=sys.stderr)
        return 2
    hits = search_product(conn, item["product_id"], args.query, BM25Cache(),
                          exclude={c["chunk_id"] for c in item["pool"]})
    print("\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits) or "（候选池外没有匹配的块）")
    return 0


def _import_drafts(conn, args) -> int:
    accepted, rejected = 0, 0
    for path in args.files:
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            qid = f"{path}:{line_no}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise LabelError("each line must be a JSON object")
                qid = str(row["qid"])
                save_label(conn, args.set, Label(qid=qid, source="draft", answerability=row["answerability"],
                                                 verdict=row.get("verdict"), grades=row["grades"],
                                                 rationale=row["rationale"], annotator=row["annotator"]))
                accepted += 1
            except (json.JSONDecodeError, KeyError, TypeError, LabelError) as exc:
                rejected += 1
                print(f"{qid}: {type(exc).__name__}: {exc}")
    print(f"accepted {accepted}, rejected {rejected}")
    return 1 if rejected else 0


def _status(conn, args) -> int:
    items = list_items(conn, args.set)
    blind_done = conn.execute("SELECT count(*) FROM annotation_labels WHERE label_set = %s AND source = 'blind'",
                              (args.set,)).fetchone()[0]
    conn.rollback()
    for split in ("dev", "test"):
        counts = {s: sum(1 for i in items if i["split"] == split and i["status"] == s)
                  for s in ("pending", "drafted", "done")}
        print(f"{split}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
              + f"; blind {sum(i['blind'] for i in items if i['split'] == split)}")
    print(f"blind labels submitted: {blind_done}")
    return 0


def _export(conn, args) -> int:
    rows = export_labels(conn, args.set)
    out_dir = args.out_root / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "labels.jsonl", rows)
    print(f"{len(rows)} final labels → {out_dir / 'labels.jsonl'}")
    return 0


def main(argv: Sequence[str] | None = None, embedder_factory: Callable[[Settings], Embedder] | None = None,
         reranker_factory: Callable[[Settings], Reranker] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gold annotation: pick questions, hand out drafts, export labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--dev", type=int, required=True)
    p.add_argument("--test", type=int, required=True)
    p.add_argument("--blind-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("packet")
    p.add_argument("--qids", required=True, help="comma-separated")
    p = sub.add_parser("search")
    p.add_argument("--qid", required=True)
    p.add_argument("--query", required=True)
    p = sub.add_parser("import-drafts")
    p.add_argument("files", nargs="+")
    sub.add_parser("status")
    sub.add_parser("export")
    for name, sp in sub.choices.items():
        sp.add_argument("--set", required=True, help="label set, e.g. gold_v1")
        if name in ("pick", "export"):
            sp.add_argument("--out-root", type=Path, default=REPO_ROOT / "annotations")
    args = parser.parse_args(argv)

    settings = load_settings()
    conn = connect(settings.database_url)
    try:
        apply_schema(conn)
        if args.command == "pick":
            return _pick(conn, args, settings, embedder_factory, reranker_factory)
        return {"packet": _packet, "search": _search, "import-drafts": _import_drafts, "status": _status,
                "export": _export}[args.command](conn, args)
    finally:
        conn.close()
