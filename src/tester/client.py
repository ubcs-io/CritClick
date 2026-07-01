"""OpenAI-compatible LLM client for vision-based game analysis.

This module provides an abstraction over any OpenAI-compatible API endpoint.
Swap the base URL to use OpenAI, Azure OpenAI, vLLM, Ollama (OpenAI compat mode),
LiteLLM, or any local proxy — without changing any other code.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod

from .models import ActionResponse

logger = logging.getLogger("tester.client")


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
    ):
        if api_key is None:
            logger.info("No API key provided — connecting without authentication.")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or "", base_url=self.api_base)

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        """Send a vision + text request to the LLM and return structured output.

        Retries on transient failures (API errors, invalid JSON) using exponential
        backoff up to ``max_retries`` times.
        """

        # Build the JSON schema from the Pydantic model so the LLM returns valid data
        response_format = self._build_response_format()

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
