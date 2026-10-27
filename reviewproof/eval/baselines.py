"""Baseline B0: the LLM answers from the category and question alone, with no reviews.
It shows how often a model answers without evidence (spec 6.5)."""
from __future__ import annotations

from pydantic import BaseModel

from reviewproof.llm import LLM, LLMResult

B0_SYSTEM = """You answer a shopper's question about a product. You are not given any reviews or product details,
only its category. Answer from general knowledge if you reasonably can; if you genuinely cannot tell for this
product, set answered to false and say so briefly."""


class B0Answer(BaseModel):
    answered: bool
    answer: str


def b0_no_evidence(llm: LLM, question: str, category: str) -> LLMResult[B0Answer]:
    return llm.structured(system=B0_SYSTEM, prompt=f"Product category: {category}\n\nQuestion:\n{question}",
                          schema=B0Answer)
