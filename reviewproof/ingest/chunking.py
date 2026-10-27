"""Split normalized review text into chunks that fit the reranker's input.

Most reviews (median 238 chars) stay whole. Longer ones are packed sentence by
sentence up to TARGET_CHARS with a one-sentence overlap, and no chunk exceeds
MAX_CHARS. Every chunk is an exact slice: review.text[start_offset:end_offset].
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from reviewproof.ingest.amazonqa import Review

MAX_CHARS = 1200
TARGET_CHARS = 800

# A sentence is a run of non-terminators followed by terminators, or a trailing
# run without one. Leading "..." becomes its own span, so no text is dropped.
_SENTENCE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    review_id: str
    product_id: str
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _sentence_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _SENTENCE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        # Split sentences longer than max_chars at the last space that fits.
        while end - start > max_chars:
            cut = text.rfind(" ", start + 1, start + max_chars + 1)
            if cut == -1:
                cut = start + max_chars
            spans.append((start, cut))
            start = cut
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            spans.append((start, end))
    return spans


def chunk_review(review: Review, max_chars: int = MAX_CHARS, target_chars: int = TARGET_CHARS) -> list[Chunk]:
    text = review.text
    if not text:
        return []
    if len(text) <= max_chars:
        bounds = [(0, len(text))]
    else:
        spans = _sentence_spans(text, max_chars)
        bounds = []
        i = 0
        while i < len(spans):
            start = spans[i][0]
            j = i
            while j + 1 < len(spans) and spans[j + 1][1] - start <= target_chars:
                j += 1
            bounds.append((start, spans[j][1]))
            if j == len(spans) - 1:
                break
            # Overlap on the last sentence only if the next chunk would still add a
            # new sentence; otherwise that chunk would repeat a single sentence.
            overlap = j > i and spans[j + 1][1] - spans[j][0] <= target_chars
            i = j if overlap else j + 1

    return [
        Chunk(
            chunk_id=f"{review.review_id}-c{index:02d}",
            review_id=review.review_id,
            product_id=review.product_id,
            chunk_index=index,
            text=text[start:end],
            start_offset=start,
            end_offset=end,
            token_estimate=estimate_tokens(text[start:end]),
        )
        for index, (start, end) in enumerate(bounds)
    ]
