"""OpenAI-compatible LLM client for vision-based game analysis.

This module provides an abstraction over any OpenAI-compatible API endpoint.
Swap the base URL to use OpenAI, Azure OpenAI, vLLM, Ollama (OpenAI compat mode),
LiteLLM, or any local proxy — without changing any other code.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod

from .models import ActionResponse, CalibrationResponse

logger = logging.getLogger("tester.client")


class HttpxUrlFormatter(logging.Formatter):
    """Scrub raw URLs from log output unless *debug_llm* is enabled.

    Operates on the fully-formatted log message so URLs are caught no matter
    how they arrive — ``%s``-style formatting, f-strings, ``record.args``,
    ``record.extra``, or any other mechanism the ``httpx`` / ``openai``
    libraries may use.

    When ``debug_llm=True`` the message is passed through unchanged.
    """

    _URL_PATTERN = re.compile(r"(https?://\S+)")

    def __init__(self, fmt=None, datefmt=None, style="%", validate=True, *, debug_llm=False):
        super().__init__(fmt, datefmt, style, validate)
        self.debug_llm = debug_llm

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if self.debug_llm:
            return formatted

        # Replace every URL in the final formatted string with just the path
        def _redact(m: re.Match) -> str:
            full_url = m.group(1)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(full_url)
                endpoint = parsed.path or "/"
            except Exception:
                endpoint = "/"
            return f"...{endpoint}"

        return self._URL_PATTERN.sub(_redact, formatted)


class LLMClient(ABC):
    """Abstract base for vision-capable LLM clients."""

    @abstractmethod
    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Send a screenshot + prompts to the LLM and return the parsed response dict.

        Args:
            image_b64: Base64-encoded PNG screenshot.
            system_prompt: System-level instruction for the model.
            user_prompt: User message describing the current task/context.

        Returns:
            A dictionary that should match the ActionResponse schema.

        Raises:
            RuntimeError: If the API call fails or returns invalid content.
        """
        ...

    @abstractmethod
    def locate_markers(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Locate numbered calibration markers in a probe image.

        Returns:
            A dictionary matching the ``CalibrationResponse`` schema.
        """
        ...


class OpenAIClient(LLMClient):
    """Client for any OpenAI-compatible API endpoint.

    Works with:
      - OpenAI API          (api_base="https://api.openai.com/v1")
      - Azure OpenAI        (api_base="https://<resource>.openai.azure.com")
      - vLLM                (api_base="http://localhost:8000/v1")
      - Ollama compat mode  (api_base="http://localhost:11434/v1")
      - LiteLLM proxy       (api_base="http://localhost:4000")
      - Any other OpenAI-compatible server
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 600,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        debug_llm: bool = False,
        reasoning_max_tokens: int | None = None,
    ):
        if api_key is None:
            logger.info("No API key provided — connecting without authentication.")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_max_tokens = reasoning_max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug_llm = debug_llm
        self._last_system_prompt: str | None = None
        self._last_user_prompt_skeleton: str | None = None

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or "", base_url=self.api_base)

    def locate_markers(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Send the calibration probe image and return located markers.

        Uses the same retrying request machinery as :meth:`analyze` but with the
        ``CalibrationResponse`` schema. Returns a dict matching that schema.
        """
        response_format = self._build_response_format(
            CalibrationResponse, name="calibration_markers"
        )
        return self._request(image_b64, system_prompt, user_prompt, response_format)

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Send a vision + text request to the LLM and return structured output.

        Retries on transient failures (API errors, invalid JSON) using exponential
        backoff up to ``max_retries`` times.
        """
        response_format = self._build_response_format()
        return self._request(image_b64, system_prompt, user_prompt, response_format)

    def _request(
        self,
        image_b64: str,
        system_prompt: str,
        user_prompt: str,
        response_format: dict,
    ) -> dict:
        """Shared vision request loop with retries and structured-JSON parsing."""
        if self.debug_llm:
            if system_prompt == self._last_system_prompt:
                logger.info(
                    "🔍 [DEBUG-LLM] --- [EXISTING SYSTEM PROMPT] (%d chars) ---",
                    len(system_prompt),
                )
            else:
                logger.info(
                    "🔍 [DEBUG-LLM] --- System prompt (%d chars) ---\n%s",
                    len(system_prompt),
                    system_prompt,
                )
                self._last_system_prompt = system_prompt
            skeleton = self._user_prompt_skeleton(user_prompt)
            if skeleton == self._last_user_prompt_skeleton:
                logger.info(
                    "🔍 [DEBUG-LLM] --- [EXISTING USER PROMPT TEMPLATE] (%d chars) ---",
                    len(user_prompt),
                )
            else:
                logger.info(
                    "🔍 [DEBUG-LLM] --- User prompt (%d chars) ---\n%s",
                    len(user_prompt),
                    user_prompt,
                )
                self._last_user_prompt_skeleton = skeleton
            logger.info(
                "🔍 [DEBUG-LLM] --- Image base64 length: %d chars ---",
                len(image_b64),
            )

        # Effective completion budget for this request. Bumped to
        # ``reasoning_max_tokens`` if a reasoning model exhausts the budget on
        # thinking tokens and returns empty content (finish_reason="length").
        effective_max_tokens = self.max_tokens

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=effective_max_tokens,
                    response_format=response_format,
                    temperature=self.temperature,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "⚠️  LLM API call attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise RuntimeError(f"LLM API call failed after {attempt} attempts: {exc}") from exc

            choice = response.choices[0]
            message = choice.message
            raw = message.content
            reasoning = self._extract_reasoning(message)
            finish_reason = choice.finish_reason

            if self.debug_llm:
                logger.info(
                    "🔍 [DEBUG-LLM] --- Raw model response (%d chars, finish_reason=%s) ---\n%s",
                    len(raw) if raw else 0,
                    finish_reason,
                    raw if raw else "(empty)",
                )
                if reasoning:
                    logger.info(
                        "🧠 [DEBUG-LLM] --- Reasoning trace (%d chars) ---\n%s",
                        len(reasoning),
                        reasoning,
                    )

            if not raw:
                # An empty ``content`` on a reasoning model usually means the
                # answer never made it out of the thinking channel — surface
                # the reasoning + finish_reason so it isn't silently opaque.
                reasoning_note = (
                    f" | reasoning: {reasoning[:200]}…" if reasoning else ""
                )
                last_exc = RuntimeError(
                    f"LLM returned empty content (finish_reason={finish_reason})."
                )
                logger.warning(
                    "⚠️  LLM returned empty content on attempt %d/%d "
                    "(finish_reason=%s)%s",
                    attempt,
                    self.max_retries + 1,
                    finish_reason,
                    reasoning_note,
                )

                # finish_reason="length" means the token budget was exhausted —
                # very likely by reasoning tokens. Retrying with the same budget
                # is futile; recover by bumping to reasoning_max_tokens if set.
                if finish_reason == "length":
                    if (
                        self.reasoning_max_tokens is not None
                        and self.reasoning_max_tokens > effective_max_tokens
                    ):
                        logger.warning(
                            "⚠️  Token budget (%d) exhausted before content was "
                            "produced (likely reasoning tokens) — retrying with "
                            "reasoning_max_tokens=%d.",
                            effective_max_tokens,
                            self.reasoning_max_tokens,
                        )
                        effective_max_tokens = self.reasoning_max_tokens
                        continue  # retry immediately with the larger budget
                    raise RuntimeError(
                        f"LLM exhausted its {effective_max_tokens}-token budget "
                        f"before producing content (finish_reason=length) — likely "
                        f"a reasoning model spending the budget on thinking. Raise "
                        f"max_tokens or set reasoning_max_tokens."
                    ) from last_exc

                if attempt <= self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exc

            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning(
                    "⚠️  LLM returned invalid JSON on attempt %d/%d: %s…",
                    attempt,
                    self.max_retries + 1,
                    raw[:200],
                )
                if attempt <= self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise RuntimeError(f"LLM returned invalid JSON after {attempt} attempts: {raw[:200]}…") from exc

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"LLM analyze failed: {last_exc}")

    @staticmethod
    def _extract_reasoning(message) -> str | None:
        """Pull a reasoning/thinking trace from non-standard message fields.

        Reasoning models served over OpenAI-compatible endpoints return their
        thinking *outside* ``message.content`` — commonly ``reasoning_content``
        (DeepSeek, vLLM, SGLang) or ``reasoning`` (OpenRouter). The stock OpenAI
        SDK doesn't model these fields, so they land in ``message.model_extra``.
        Returns the first non-empty trace found, else ``None``.
        """
        extra = getattr(message, "model_extra", None) or {}
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(message, attr, None)
            if val is None:
                val = extra.get(attr)
            if isinstance(val, str) and val.strip():
                return val
        return None

    @staticmethod
    def _user_prompt_skeleton(user_prompt: str) -> str:
        """Replace the variable context block with a placeholder for comparison.

        The user prompt template has a ``{context}`` slot that gets filled with
        per-step narrative context.  This helper strips that section so the
        static wrapper is all that remains, allowing us to detect when the
        template itself has changed vs. when only the context differs.
        """
        import re

        return re.sub(
            r"Recent narrative context:\n.*?\n\nIdentify interactive elements",
            "Recent narrative context:\n[...]\n\nIdentify interactive elements",
            user_prompt,
            flags=re.DOTALL,
        )

    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff: delay * 2^(attempt-1)."""
        delay = self.retry_delay * (2 ** (attempt - 1))
        logger.info("⏳ Retrying in %.1fs (exponential backoff)…", delay)
        time.sleep(delay)

    def _build_response_format(
        self, model: type = ActionResponse, name: str = "game_action"
    ) -> dict:
        """Build an OpenAI `response_format` dict from a Pydantic model's schema."""
        schema = model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }
