"""Server-sent events for one question.

Progress events stream live, one per finished workflow node. The answer is published in chunks
only after the workflow has verified its citations: this is not token-by-token generation.
The workflow runs in its own thread under the services lock and always finishes and saves its
run, even when the client disconnects halfway.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import asdict, dataclass
from typing import Iterator

from reviewproof.api.services import Services
from reviewproof.runs import new_run_id, save_run
from reviewproof.workflow.graph import QAResult, stream_answer
from reviewproof.workflow.state import TraceEntry

log = logging.getLogger(__name__)
CHUNK_CHARS = 120


@dataclass(frozen=True)
class _Finished:
    result: QAResult
    persisted: bool


def answer_chunks(text: str) -> list[str]:
    """Split by line (keeping line ends), then cut long lines; joined, they give back the text."""
    chunks = []
    for line in text.splitlines(keepends=True):
        chunks += [line[i:i + CHUNK_CHARS] for i in range(0, len(line), CHUNK_CHARS)]
    return chunks


def _run_workflow(services: Services, run_id: str, product_id: str, question: str, out: queue.Queue) -> None:
    with services.lock:
        conn = None
        try:
            conn = services.connect()
            result = None
            for item in stream_answer(conn, product_id, question, llm=services.llm, retrieval=services.retrieval,
                                      max_rewrites=services.settings.max_rewrites):
                if isinstance(item, TraceEntry):
                    out.put(item)
                else:
                    result = item
            try:
                save_run(conn, run_id, result, services.llm.model_name)
                persisted = True
            except Exception:
                log.exception("could not save run %s", run_id)
                persisted = False
            out.put(_Finished(result, persisted))
        except Exception as exc:  # unexpected: the workflow itself raised
            log.exception("run %s failed", run_id)
            out.put(exc)
        finally:
            if conn is not None:
                conn.close()


def qa_events(services: Services, product_id: str, question: str) -> Iterator[tuple[str, dict]]:
    run_id = new_run_id()
    yield "run_started", {"run_id": run_id, "product_id": product_id, "question": question}
    out: queue.Queue = queue.Queue()
    threading.Thread(target=_run_workflow, args=(services, run_id, product_id, question, out),
                     name=f"qa-{run_id}", daemon=True).start()
    while True:
        item = out.get()
        if isinstance(item, TraceEntry):
            yield "progress", asdict(item)
        elif isinstance(item, Exception):
            yield "error", {"run_id": run_id, "message": f"{type(item).__name__}: {item}"}
            return
        else:
            r = item.result
            for chunk in answer_chunks(r.answer_text):
                yield "answer_chunk", {"text": chunk}
            yield "citations", {"citations": [asdict(c) for c in r.citations]}
            yield "done", {
                "run_id": run_id, "status": r.status, "service_status": r.service_status, "verdict": r.verdict,
                "route": r.route, "unknowns": list(r.unknowns), "search_queries": list(r.search_queries),
                "retrieval_modes": list(r.retrieval_modes), "rewrites_used": r.rewrites_used,
                "llm_calls": r.llm_calls, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms, "persisted": item.persisted}
            return
