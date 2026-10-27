import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx2
from pydantic import BaseModel, ValidationError

from reviewproof.llm import ClaudeLLM, LLMError
from tests.support import FakeLLM


class Answer(BaseModel):
    text: str


def response(parsed, stop_reason="end_turn", usage=(120, 30)):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]))


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: records messages.parse kwargs."""

    def __init__(self, reply=None, error=None):
        self.messages = self
        self.reply, self.error, self.calls = reply, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply


class ClaudeLLMTests(unittest.TestCase):
    def test_parse_call_shape(self):
        fake = FakeAnthropic(response(Answer(text="hi")))
        result = ClaudeLLM(client=fake).structured(system="sys", prompt="q", schema=Answer)
        self.assertEqual((result.output.text, result.input_tokens, result.output_tokens), ("hi", 120, 30))
        self.assertGreaterEqual(result.latency_ms, 0)
        kwargs = fake.calls[0]
        self.assertEqual(kwargs["model"], "claude-haiku-5-5")
        self.assertEqual(kwargs["system"], "sys")
        self.assertEqual(kwargs["output_format"], Answer)
        self.assertEqual(kwargs["output_config"], {"effort": "low"})
        self.assertEqual(kwargs["max_tokens"], 4096)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "q"}])
        for banned in ("temperature", "top_p", "thinking"):
            self.assertNotIn(banned, kwargs)

    def test_model_and_effort(self):
        fake = FakeAnthropic(response(Answer(text="hi")))
        ClaudeLLM(model="m", effort="medium", client=fake).structured(system="s", prompt="p", schema=Answer)
        self.assertEqual((fake.calls[0]["model"], fake.calls[0]["output_config"]), ("m", {"effort": "medium"}))

    def test_refusal_truncation_and_missing_output_raise(self):
        cases = [(response(None, "refusal"), "refused"), (response(Answer(text="x"), "max_tokens"), "truncated"),
                 (response(None), "no structured output")]
        for reply, message in cases:
            with self.assertRaisesRegex(LLMError, message):
                ClaudeLLM(client=FakeAnthropic(reply)).structured(system="s", prompt="p", schema=Answer)

    def test_sdk_errors_become_llm_error(self):
        error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        with self.assertRaisesRegex(LLMError, "APIConnectionError"):
            ClaudeLLM(client=FakeAnthropic(error=error)).structured(system="s", prompt="p", schema=Answer)

    def test_schema_validation_error_becomes_llm_error(self):
        # messages.parse validates eagerly; a reply cut off mid-JSON raises pydantic's ValidationError.
        try:
            Answer.model_validate_json('{"text": "cut of')
        except ValidationError as exc:
            error = exc
        with self.assertRaisesRegex(LLMError, "invalid structured output"):
            ClaudeLLM(client=FakeAnthropic(error=error)).structured(system="s", prompt="p", schema=Answer)

    def test_missing_key_is_llm_error_not_type_error(self):
        # Without credentials the SDK raises a bare TypeError at request time, outside AnthropicError.
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(LLMError, "ANTHROPIC_API_KEY"):
                ClaudeLLM().structured(system="s", prompt="p", schema=Answer)


class FakeLLMTests(unittest.TestCase):
    def test_script(self):
        llm = FakeLLM({"Answer": [Answer(text="a"), lambda prompt: Answer(text=prompt), LLMError("boom")]})
        self.assertEqual(llm.structured(system="s", prompt="p", schema=Answer).output.text, "a")
        self.assertEqual(llm.structured(system="s", prompt="echo", schema=Answer).output.text, "echo")
        with self.assertRaises(LLMError):
            llm.structured(system="s", prompt="p", schema=Answer)
        with self.assertRaises(AssertionError):
            llm.structured(system="s", prompt="p", schema=Answer)
        self.assertEqual([c[0] for c in llm.calls], ["Answer"] * 4)


if __name__ == "__main__":
    unittest.main()
