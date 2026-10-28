"""Pure post-processing: keep only citations of this run's candidates, settle the final status,
and render the answer text with fixed copy in the question's language (Chinese or English)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from reviewproof.retrieval.types import Candidate
from reviewproof.workflow.schemas import Claim

_CJK = re.compile(r"[一-鿿]")

COPY = {
    "zh": {
        "service_failed": "服务出错，这次没有完成检索或评估，不代表评论里没有答案。",
        "out_of_scope": "这个问题不在评论能回答的范围内：{reason}",
        "unanswerable": "评论无法确认这个问题。",
        "unknowns": "评论没有提到：{items}",
        "separator": "；",
        "verdict": "结论：{label}",
        "labels": {"yes": "是", "no": "否", "mixed": "评论意见不一", "unknown": "无法确认"},
    },
    "en": {
        "service_failed": "A service error stopped this run; this is not a finding about the reviews.",
        "out_of_scope": "This question is outside what reviews can answer: {reason}",
        "unanswerable": "The reviews don't confirm this.",
        "unknowns": "Not covered by the reviews: {items}",
        "separator": "; ",
        "verdict": "Verdict: {label}",
        "labels": {"yes": "Yes", "no": "No", "mixed": "Reviews disagree", "unknown": "Can't confirm"},
    },
}


@dataclass(frozen=True)
class Citation:
    number: int  # from 1, in order of first appearance
    chunk_id: str
    review_id: str  # system-generated, not an Amazon review ID
    text: str


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    citation_numbers: tuple[int, ...]


def is_chinese(text: str) -> bool:
    return bool(_CJK.search(text))


def verify_claims(
    claims: Sequence[Claim], candidates: Sequence[Candidate], product_id: str
) -> tuple[tuple[AnswerClaim, ...], tuple[Citation, ...], tuple[str, ...]]:
    allowed = {c.chunk.chunk_id: c.chunk for c in candidates if c.chunk.product_id == product_id}
    numbers: dict[str, int] = {}
    citations: list[Citation] = []
    kept: list[AnswerClaim] = []
    dropped: list[str] = []
    for claim in claims:
        cited: list[int] = []
        for chunk_id in claim.chunk_ids:
            if chunk_id not in allowed:
                dropped.append(chunk_id)
                continue
            if chunk_id not in numbers:
                numbers[chunk_id] = len(numbers) + 1
                chunk = allowed[chunk_id]
                citations.append(Citation(numbers[chunk_id], chunk_id, chunk.review_id, chunk.text))
            if numbers[chunk_id] not in cited:
                cited.append(numbers[chunk_id])
        if cited:
            kept.append(AnswerClaim(claim.text, tuple(cited)))
    return tuple(kept), tuple(citations), tuple(dict.fromkeys(dropped))


def final_status(assessed: str, claims: Sequence[AnswerClaim]) -> str:
    """The evidence assessment decides answerability; verification can only take it down to
    unanswerable when no cited claim survives."""
    return "unanswerable" if assessed == "unanswerable" or not claims else assessed


def render_answer(*, question: str, route: str, out_of_scope_reason: str, service_status: str,
                  status: str | None, verdict: str | None, claims: Sequence[AnswerClaim],
                  unknowns: Sequence[str]) -> str:
    copy = COPY["zh" if is_chinese(question) else "en"]
    if service_status == "failed":
        return copy["service_failed"]
    if route == "out_of_scope":
        return copy["out_of_scope"].format(reason=out_of_scope_reason)
    lines = []
    if verdict is not None:
        lines.append(copy["verdict"].format(label=copy["labels"][verdict]))
    if status == "unanswerable":
        lines.append(copy["unanswerable"])
        return "\n".join(lines)
    lines += [f"{c.text} " + "".join(f"[{n}]" for n in c.citation_numbers) for c in claims]
    if unknowns:
        lines.append(copy["unknowns"].format(items=copy["separator"].join(unknowns)))
    return "\n".join(lines)
