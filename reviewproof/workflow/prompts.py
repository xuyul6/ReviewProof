"""Prompts for the three LLM steps.

Only two things ever reach a prompt: the question being asked and the current product's review
chunks. Community answers, snippets, labels and other questions from the `questions` table never do.
"""
from __future__ import annotations

from typing import Sequence

from reviewproof.retrieval.types import Candidate

PLAN_SYSTEM = """You plan searches for a pre-purchase QA assistant that answers questions about one product \
using only that product's customer reviews.

Decide the route:
- "out_of_scope" when reviews cannot or should not answer it: price, stock, shipping or delivery, \
promotions, warranty or return policy, official specifications that must come from the seller, \
comparisons with or recommendations of other products, or definitive medical, safety or legal \
judgements. Give a short reason in the question's language.
- otherwise "review_question", with an empty out_of_scope_reason.

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
missing: the parts of the question the excerpts do not cover, written in the question's language.
rewrite_query: if something is missing, an English search query aimed at the missing part; else empty."""

GENERATE_SYSTEM = """You answer a shopper's question about one product using only the given review excerpts.

- Write each finding as a separate claim, in the question's language, and cite at least one excerpt \
id for it in chunk_ids, copied exactly from the brackets. Do not state anything the excerpts do not support.
- When reviewers disagree, give each side as its own claim instead of choosing one, and use verdict "mixed".
- List parts of the question the excerpts do not cover in unknowns, in the question's language.
- verdict only for yes/no questions: "yes", "no", "mixed", or "unknown"; otherwise null."""


def _excerpts(candidates: Sequence[Candidate]) -> str:
    return "\n\n".join(f"[{c.chunk.chunk_id}] {c.chunk.text}" for c in candidates)


def plan_prompt(question: str) -> str:
    return f"Question:\n{question}"


def assess_prompt(question: str, candidates: Sequence[Candidate]) -> str:
    return f"Question:\n{question}\n\nReview excerpts:\n{_excerpts(candidates)}"


def generate_prompt(question: str, is_yes_no: bool, status: str, candidates: Sequence[Candidate],
                    missing: Sequence[str]) -> str:
    lines = [f"Question:\n{question}", f"Yes/no question: {'yes' if is_yes_no else 'no'}",
             f"Assessment: {status}"]
    if missing:
        lines.append("Not covered by the excerpts: " + "; ".join(missing))
    lines.append(f"Review excerpts:\n{_excerpts(candidates)}")
    return "\n\n".join(lines)
