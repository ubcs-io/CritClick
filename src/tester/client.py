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

from .models import ActionResponse

logger = logging.getLogger("tester.client")


class HttpxUrlFilter(logging.Filter):
    """Scrub raw URLs from ``httpx`` log records unless *debug_llm* is enabled.

    By default the ``httpx`` library logs the full request URL (e.g.
    ``HTTP Request: POST https://api.openai.com/v1/chat/completions …``).
    This filter replaces the URL with a generic endpoint path so only the
    API operation is visible, keeping API keys that may be embedded in the
    query string out of plain-text logs.

    When ``debug_llm=True`` the record is passed through unchanged.
    """

    _URL_PATTERN = re.compile(r"(https?://\S+)")

    def __init__(self, debug_llm: bool = False):
        super().__init__()
        self.debug_llm = debug_llm

    def filter(self, record: logging.LogRecord) -> bool:
        if self.debug_llm:
            return True
        match = self._URL_PATTERN.search(record.getMessage())
        if match:
            full_url = match.group(1)
            # Extract the path portion so the endpoint type is still visible
            try:
                from urllib.parse import urlparse
                parsed = urlparse(full_url)
                endpoint = parsed.path or "/"
            except Exception:
                endpoint = "/"
            redacted = f"...{endpoint}"

            # Scrub from record.msg (handles f-string / direct format style)
            if isinstance(record.msg, str):
                record.msg = record.msg.replace(full_url, redacted)

            # Scrub from record.args (handles %s-style format where the URL
            # lives in positional arguments — this is how httpx logs).
            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        a.replace(full_url, redacted) if isinstance(a, str) else a
                        for a in record.args
                    )
                elif isinstance(record.args, dict):
                    record.args = {
                        k: v.replace(full_url, redacted) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, str):
                    record.args = record.args.replace(full_url, redacted)
        return True


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
    ):
        if api_key is None:
            logger.info("No API key provided — connecting without authentication.")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug_llm = debug_llm
        self._last_system_prompt: str | None = None
        self._last_user_prompt_skeleton: str | None = None

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or "", base_url=self.api_base)

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Send a vision + text request to the LLM and return structured output.

        Retries on transient failures (API errors, invalid JSON) using exponential
        backoff up to ``max_retries`` times.
        """

        # Build the JSON schema from the Pydantic model so the LLM returns valid data
        response_format = self._build_response_format()

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
                    max_tokens=self.max_tokens,
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

            raw = response.choices[0].message.content

            if self.debug_llm:
                logger.info(
                    "🔍 [DEBUG-LLM] --- Raw model response (%d chars) ---\n%s",
                    len(raw) if raw else 0,
                    raw if raw else "(empty)",
                )

            if not raw:
                last_exc = RuntimeError("LLM returned empty content.")
                logger.warning(
                    "⚠️  LLM returned empty content on attempt %d/%d",
                    attempt,
                    self.max_retries + 1,
                )
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

    def _build_response_format(self) -> dict:
        """Build an OpenAI `response_format` dict from the ActionResponse schema."""
        schema = ActionResponse.model_json_schema()
        # Remove top-level schema keys that OpenAI doesn't expect
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "game_action",
                "strict": True,
                "schema": schema,
            },
        }
