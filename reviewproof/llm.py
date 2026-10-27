"""Claude structured output: one Pydantic schema in, one validated instance out.

Every failure (missing key, API or connection error, refusal, truncation) becomes LLMError,
so workflow nodes have a single exception to turn into a service failure.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 4096  # Haiku 5.5 thinks by default, and thinking counts toward this


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    output: T
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLM(Protocol):
    model_name: str

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]: ...


class ClaudeLLM:
    """Reads ANTHROPIC_API_KEY through the SDK on first use; never stores or prints it."""

    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "low", client=None, timeout: float = 60.0):
        self.model_name = model
        self.effort = effort
        self.timeout = timeout
        self._client = client

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        start = time.perf_counter()
        try:
            if self._client is None:
                # The SDK reports missing credentials as a bare TypeError at request time; check first.
                if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                    raise LLMError("ANTHROPIC_API_KEY is not set")
                self._client = anthropic.Anthropic(timeout=self.timeout)
            response = self._client.messages.parse(
                model=self.model_name,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                output_config={"effort": self.effort},
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("refused")
        if response.stop_reason == "max_tokens":
            raise LLMError("truncated")
        if response.parsed_output is None:
            raise LLMError("no structured output")
        return LLMResult(response.parsed_output, response.usage.input_tokens, response.usage.output_tokens,
                         (time.perf_counter() - start) * 1000)
