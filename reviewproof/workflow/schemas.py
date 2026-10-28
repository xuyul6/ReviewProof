"""Structured outputs the LLM must return at each workflow step."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class QueryPlan(BaseModel):
    route: Literal["review_question", "out_of_scope"]
    out_of_scope_reason: str  # empty when route is review_question
    is_yes_no: bool
    search_query: str  # English; Chinese questions are translated here


class EvidenceAssessment(BaseModel):
    status: Literal["answered", "partial", "unanswerable"]
    supporting_chunk_ids: list[str]
    missing: list[str]  # parts of the question the reviews don't cover, in the answer language
    rewrite_query: str  # English search query aimed at what is missing; empty if none


class Claim(BaseModel):
    text: str  # one finding, in the answer language (Chinese or English)
    chunk_ids: list[str]


class GeneratedAnswer(BaseModel):
    claims: list[Claim]
    unknowns: list[str]
    # Only for yes/no questions. Generation runs only for answered or partial, so "unknown" (which goes
    # with unanswerable) is not offered.
    verdict: Literal["yes", "no", "mixed"] | None
