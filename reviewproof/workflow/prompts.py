"""Prompts for the three LLM steps.

Only two things ever reach a prompt: the question being asked and the current product's review
chunks. Community answers, snippets, labels and other questions from the `questions` table never do.
"""
from __future__ import annotations

from typing import Sequence

from reviewproof.retrieval.types import Candidate
from reviewproof.workflow.answer import is_chinese

PLAN_SYSTEM = """You plan searches for a pre-purchase QA assistant that answers questions about one product \
using only that product's customer reviews.

Decide the route:
- "out_of_scope" only when no part of the question is about using or owning this product: price, \
stock, shipping or delivery, promotions, questions to the seller about their own business ("do you \
carry...", "do you ship to..."), requests to recommend other products, or definitive medical or legal \
judgements. Give a short reason in the answer language.
- otherwise "review_question", with an empty out_of_scope_reason. This includes warranty, official \
specifications and dimensions, compatibility or fit, authenticity, and comparisons of this product \
with others: reviewers often report their own experience of these, and the next step checks whether \
they do.

Set is_yes_no to true when the question expects a yes or no answer.
Write search_query in English (translate if the question is not in English): a short query with the \
words a reviewer would use, for searching the reviews."""

ASSESS_SYSTEM = """You judge whether a product's customer reviews answer a shopper's question. Use only the \
review excerpts given; do not fill gaps with general knowledge or assumptions about the product.

status:
- "answered": the excerpts answer every part of the question.
- "partial": they answer some parts; list the uncovered parts in missing.
- "unanswerable": they contain no information that answers the question.

supporting_chunk_ids: ids of the excerpts that support an answer, copied exactly from the brackets.
missing: the parts of the question the excerpts do not cover, written in the answer language.
rewrite_query: if something is missing, an English search query aimed at the missing part; else empty."""

GENERATE_SYSTEM = """You answer a shopper's question about one product using only the given review excerpts.

- Write each finding as a separate claim, in the answer language, and cite at least one excerpt \
id for it in chunk_ids, copied exactly from the brackets. Do not state anything the excerpts do not support.
- A claim reports what reviewers say. That the excerpts do not mention something goes in unknowns, \
never as a claim.
- When reviewers disagree, give each side as its own claim instead of choosing one, and use verdict "mixed".
- List parts of the question the excerpts do not cover in unknowns, in the answer language.
- verdict only for yes/no questions, otherwise null. It answers the first yes/no question asked, as \
worded, which yes/no answers copy into verdict_question first ("Does it come with X?" when X is sold \
separately is "no"); cover any follow-up parts in claims. \
Use "yes" or "no" from what the excerpts show, or \
"mixed" when reviewers disagree. A close case (another model year, a similar size) still leans yes or \
no; say in the claim that it is a close case. Use "unknown" only when the excerpts lean neither way \
(a reviewer's worry or guess is not evidence); the answer is then reported as not confirmed."""


def _excerpts(candidates: Sequence[Candidate]) -> str:
    return "\n\n".join(f"[{c.chunk.chunk_id}] {c.chunk.text}" for c in candidates)


def _language(question: str) -> str:
    # The fixed copy exists only in Chinese and English (answer.py), so model text must match it.
    return f"Answer language: {'Chinese' if is_chinese(question) else 'English'}"


def plan_prompt(question: str) -> str:
    return f"Question:\n{question}\n\n{_language(question)}"


def assess_prompt(question: str, candidates: Sequence[Candidate]) -> str:
    return f"Question:\n{question}\n\n{_language(question)}\n\nReview excerpts:\n{_excerpts(candidates)}"


def generate_prompt(question: str, is_yes_no: bool, status: str, candidates: Sequence[Candidate],
                    missing: Sequence[str]) -> str:
    lines = [f"Question:\n{question}", _language(question), f"Yes/no question: {'yes' if is_yes_no else 'no'}",
             f"Assessment: {status}"]
    if missing:
        lines.append("Not covered by the excerpts: " + "; ".join(missing))
    lines.append(f"Review excerpts:\n{_excerpts(candidates)}")
    return "\n\n".join(lines)
