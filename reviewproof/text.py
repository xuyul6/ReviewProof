from __future__ import annotations

import html
import re

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form for stored review text, chunking and snippet alignment.

    Review texts and AmazonQA snippets both contain HTML entities (`&#39;`, `&quot;`),
    so every comparison between them must go through this function.
    """
    return _WHITESPACE.sub(" ", html.unescape(text)).strip()


def dedupe_key(text: str) -> str:
    return normalize_text(text).lower()
