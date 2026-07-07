"""Tests for the OpenAI-compatible client: reasoning extraction + recovery."""

import types

import pytest

from tester.client import OpenAIClient


def _make_client(**kwargs) -> OpenAIClient:
    """Build a client without hitting the network (no key needed)."""
    return OpenAIClient(api_base="http://localhost:11434/v1", api_key="", **kwargs)


def _msg(**attrs) -> types.SimpleNamespace:
    """A stub message. model_extra defaults to {} unless overridden."""
    attrs.setdefault("model_extra", {})
    return types.SimpleNamespace(**attrs)


def _response(content, finish_reason, **msg_attrs):
    """Stub an OpenAI chat.completions response object."""
    message = _msg(content=content, **msg_attrs)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice])


class TestExtractReasoning:
    def test_reasoning_content_attribute(self):
        msg = _msg(content=None, reasoning_content="I am thinking.")
        assert OpenAIClient._extract_reasoning(msg) == "I am thinking."

    def test_reasoning_attribute(self):
        msg = _msg(content=None, reasoning="OpenRouter style.")
        assert OpenAIClient._extract_reasoning(msg) == "OpenRouter style."

    def test_reads_from_model_extra(self):
        # The stock OpenAI SDK routes unknown fields into model_extra.
        msg = _msg(content=None, model_extra={"reasoning_content": "extra field"})
        assert OpenAIClient._extract_reasoning(msg) == "extra field"

    def test_reasoning_content_preferred_over_reasoning(self):
        msg = _msg(content=None, reasoning_content="first", reasoning="second")
        assert OpenAIClient._extract_reasoning(msg) == "first"

    def test_none_when_absent(self):
        assert OpenAIClient._extract_reasoning(_msg(content="{}")) is None

    def test_none_when_blank(self):
        msg = _msg(content=None, reasoning_content="   ")
        assert OpenAIClient._extract_reasoning(msg) is None


class TestLengthRecovery:
    def _run(self, client, responses):
        """Stub the SDK create() to yield *responses* in order, recording kwargs."""
        calls: list[dict] = []

        def fake_create(**kwargs):
            calls.append(kwargs)
            return responses[len(calls) - 1]

        client._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        )
        return calls

    def test_bumps_budget_on_length_truncation(self):
        client = _make_client(max_tokens=600, reasoning_max_tokens=4096, max_retries=3)
        responses = [
            # 1st: budget exhausted by thinking, empty content.
            _response(None, "length", reasoning_content="thinking hard"),
            # 2nd: with the larger budget, real JSON comes out.
            _response('{"ok": true}', "stop"),
        ]
        calls = self._run(client, responses)

        result = client._request("img", "sys", "user", {"type": "json_object"})
        assert result == {"ok": True, "_reasoning_chars": 0}
        assert [c["max_tokens"] for c in calls] == [600, 4096]

    def test_fails_fast_without_reasoning_budget(self):
        client = _make_client(max_tokens=600, reasoning_max_tokens=None, max_retries=3)
        responses = [_response(None, "length", reasoning_content="thinking")]
        calls = self._run(client, responses)

        with pytest.raises(RuntimeError, match="finish_reason=length"):
            client._request("img", "sys", "user", {"type": "json_object"})
        # No futile identical retries — a single call, then fail fast.
        assert len(calls) == 1
