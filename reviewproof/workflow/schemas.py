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
    verdict: Literal["yes", "no", "mixed"] | None  # null; yes/no questions use GeneratedYesNoAnswer


class GeneratedYesNoAnswer(BaseModel):
    """For yes/no questions. Field order is the order the model writes: first the yes/no question the
    verdict answers, last the verdict, which is required. "unknown" means the excerpts lean neither
    way; verification then makes the run unanswerable, so unknown never goes with answered or partial."""
    verdict_question: str  # the first yes/no question asked, as worded
    claims: list[Claim]
    unknowns: list[str]
    verdict: Literal["yes", "no", "mixed", "unknown"]


class GeneralNote(BaseModel):
    """Shown under a "not from the reviews" heading when the reviews can't answer (all or part)."""
    kind: Literal["general", "suggestion"]  # what is typical for this kind of product / where to check
    text: str
