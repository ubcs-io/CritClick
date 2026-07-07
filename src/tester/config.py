"""Configuration loading from TOML files with environment variable overrides.

Resolution order (last wins):
  1. Defaults hardcoded in the Settings model
  2. Values loaded from a TOML config file
  3. Environment variables (e.g. TESTER_API_KEY, TESTER_API_BASE, …)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import GameConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_ip(hostname: str) -> "ipaddress.IPv4Address | ipaddress.IPv6Address | None":
    """Resolve *hostname* to an IP address, or return ``None`` on failure."""
    import ipaddress

    try:
        import socket
    except ImportError:
        return None

    try:
        addr = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        if addr:
            return ipaddress.ip_address(addr[0][4][0])
    except (socket.gaierror, ValueError, OSError):
        return None
    return None

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

class LLMSettings(BaseModel):
    """Configuration for the OpenAI-compatible LLM endpoint."""

    api_base: str = Field(
        default="https://api.openai.com/v1",
        description=(
            "Base URL for the OpenAI-compatible API. "
            "Change to point at any compatible endpoint: Azure, vLLM, Ollama, LiteLLM, etc."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "API key for the endpoint. If omitted, the client will look for the "
            "OPENAI_API_KEY environment variable, or connect without auth (for local endpoints)."
        ),
    )
    model: str = Field(
        default="gpt-4o",
        description="Model identifier (e.g. 'gpt-4o', 'gpt-4o-mini', 'llava-v1.6-mistral-7b').",
    )
    max_tokens: int = Field(
        default=600, ge=1, le=16384,
        description=(
            "Max tokens for the completion. For reasoning models (o1, o3, "
            "deepseek-r1) this is the TOTAL budget for thinking + output "
            "combined. Set low (e.g. 600) to cap thinking time; raise only "
            "if the model consistently returns empty content."
        ),
    )
    reasoning_max_tokens: int | None = Field(
        default=None, ge=1, le=16384,
        description=(
            "Larger completion budget used to recover when a reasoning/thinking "
            "model exhausts max_tokens on its thinking and returns empty content "
            "(finish_reason='length'). Only relevant for reasoning models; leave "
            "unset for standard models."
        ),
    )
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Thinking effort level for o-series reasoning models (o1, o3). "
            "One of 'low', 'medium', 'high'. 'low' dramatically reduces thinking "
            "token usage for simple UI/click tasks. When unset and the model name "
            "starts with 'o1' or 'o3', defaults to 'low' automatically. "
            "Ignored by standard models like gpt-4o."
        ),
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout: float | None = Field(
        default=60.0, ge=1.0,
        description=(
            "Per-request timeout in seconds. When set, a call that exceeds this "
            "duration is terminated and retried. Caps runaway thinking on "
            "reasoning models. Defaults to 60 s; set to null (None) to disable "
            "the timeout entirely."
        ),
    )
    extra_body: dict | None = Field(
        default=None,
        description=(
            "Additional parameters to pass directly in the API request body. "
            "Use this for model-specific options not exposed as top-level config "
            "(e.g. vLLM 'thinking' settings: {'chat_template_kwargs': {'thinking': False}})."
        ),
    )
    max_retries: int = Field(
        default=3, ge=0,
        description="Maximum number of retries for transient LLM failures (rate limits, timeouts).",
    )
    retry_delay: float = Field(
        default=2.0, ge=0.0,
        description="Base delay in seconds for exponential backoff between retries.",
    )
    max_completion_tokens: int | None = Field(
        default=None, ge=1, le=16384,
        description=(
            "If supported by the provider, caps only the visible output tokens "
            "(not thinking/reasoning tokens). This is the preferred mechanism "
            "for reasoning models because it prevents thinking from stealing "
            "the entire max_tokens budget. When None, max_tokens is used as a "
            "combined cap for thinking + output. Set this to a small value "
            "(e.g. 300) to enforce short answers regardless of thinking depth."
        ),
    )
    min_reasoning_ratio: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description=(
            "When the ratio of reasoning tokens to the max_tokens budget exceeds "
            "this threshold, the harness tightens max_tokens for subsequent "
            "calls (adaptive throttling). Set to 1.0 to disable throttling. "
            "Only relevant for reasoning models that return thinking traces."
        ),
    )
    block_external_routing: bool = Field(
        default=False,
        description=(
            "If True, the LLM endpoint must resolve to a private/loopback IP "
            "(e.g. localhost, 127.0.0.1, 10.x.x.x, 192.168.x.x). "
            "Useful for air-gapped deployments that should not send data "
            "to cloud inference providers like OpenAI."
        ),
    )

    @field_validator("block_external_routing", mode="after")
    @classmethod
    def _validate_endpoint_when_blocked(cls, blocked: bool, info: "ValidationInfo") -> bool:
        """When ``block_external_routing`` is True, ensure the endpoint is private.

        Resolves the hostname via DNS and checks whether the resulting IP
        is private (RFC 1918, loopback, or link-local).  Falls back to a
        fast string-prefix check when DNS resolution fails so that tests
        and unusual hostnames aren't silently allowed through.
        """
        if not blocked:
            return blocked

        api_base = info.data.get("api_base") if info.data else None
        if not api_base:
            return blocked

        from urllib.parse import urlparse

        parsed = urlparse(api_base)
        hostname = parsed.hostname or ""

        # Fast path: well-known loopback literals need no DNS lookup.
        if hostname in ("localhost", "127.0.0.1", "::1", "[::1]"):
            return blocked

        # Try full DNS resolution first.
        ip = _resolve_ip(hostname)
        if ip is not None:
            if ip.is_private:
                return blocked
            # Public IP — fall through to error.

        # Fallback: check for common private prefixes (covers cases where
        # DNS resolution failed but the user wrote a private IP directly).
        import ipaddress as _ipmod
        try:
            candidate = _ipmod.ip_address(hostname)
            if candidate.is_private:
                return blocked
        except ValueError:
            pass

        # Explicit failure for the common case.
        raise ValueError(
            f"block_external_routing=True but endpoint '{api_base}' "
            f"(hostname '{hostname}') resolves to a public IP. "
            f"Use a local endpoint such as http://localhost:11434/v1."
        )


# ---------------------------------------------------------------------------
# Harness runtime settings
# ---------------------------------------------------------------------------

class HarnessSettings(BaseModel):
    """Controls the playthrough loop behaviour."""

    max_steps: int = Field(default=50, ge=1, description="Max LLM calls per run.")
    wait_after_action: float = Field(
        default=1.5, ge=0.0,
        description="Seconds to pause after executing an action (lets the game respond).",
    )
    stuck_threshold: int = Field(
        default=3, ge=1,
        description="Number of repeated identical actions before the harness warns.",
    )
    screen_scale: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Multiplier for LLM-returned coordinates (handles DPI mismatches).",
    )
    coordinate_grid: bool = Field(
        default=False,
        description="If True, overlay a labeled coordinate grid on screenshots "
        "sent to the model so it can read click coordinates off reference lines. "
        "Usually unnecessary for models that ground natively (e.g. Qwen-VL); "
        "prefer coordinate calibration instead.",
    )
    grid_spacing: int = Field(
        default=100, ge=20, le=500,
        description="Pixel spacing between gridlines when coordinate_grid is enabled.",
    )
    coordinate_calibration: bool = Field(
        default=True,
        description="If True, run a one-time calibration probe at startup to learn "
        "how the model's returned coordinates map to image pixels (handles models "
        "that emit normalized/resized coordinates, e.g. Qwen-VL's 0-1000 space).",
    )
    coordinate_max: int = Field(
        default=0, ge=0,
        description="Fallback for the model's coordinate convention when calibration "
        "is disabled or fails. 0 = raw pixels; e.g. 1000 = normalized 0-1000 per axis.",
    )
    headless: bool = Field(
        default=False,
        description="If True, attempt to run on a virtual display (Linux + Xvfb).",
    )
    wait_duration: float = Field(
        default=2.0, ge=0.0,
        description="Seconds to pause when the LLM returns a 'wait' action.",
    )
    startup_countdown: bool = Field(
        default=True,
        description="If True, print a 3-second countdown before launching the game.",
    )
    stuck_recovery: bool = Field(
        default=True,
        description="If True, attempt automatic recovery strategies when stuck state is detected.",
    )
    window_find_timeout: float = Field(
        default=15.0, ge=0.0, le=120.0,
        description="Maximum seconds to wait for the game window to appear after launch.",
    )


# ---------------------------------------------------------------------------
# Logging / output settings
# ---------------------------------------------------------------------------

class LoggingSettings(BaseModel):
    """Controls what and how the harness records during a playthrough."""

    log_file: str = Field(default="playthrough_log.jsonl")
    save_screenshots: bool = Field(default=True)
    screenshot_dir: str = Field(default="screenshots")
    step_history_window: int = Field(
        default=8, ge=0,
        description="Number of recent step-history entries to include in the LLM prompt. "
        "Each entry records the action taken and where it clicked. Set high enough to span "
        "a full game-loop cycle so the model remembers which screens it has already visited.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


# ---------------------------------------------------------------------------
# Review / recap settings
# ---------------------------------------------------------------------------

class ReviewSettings(BaseModel):
    """Controls the end-of-run, persona-driven player-experience review."""

    personas: list[str] = Field(
        default_factory=list,
        description=(
            "Player personas to review each run from. Each becomes one section in "
            "the experience review, written in that persona's voice (e.g. "
            "'Impatient completionist who hates dead ends', 'First-time casual who "
            "is easily confused'). When empty, the run is reviewed as a single "
            "'Typical player'."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Complete tester configuration, loadable from TOML + environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TESTER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    game: GameConfig
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)

    # Prompt overrides (optional — use defaults if omitted)
    system_prompt: str | None = Field(
        default=None,
        description="Override the default system prompt for the LLM.",
    )
    user_prompt_template: str | None = Field(
        default=None,
        description="Override the default user prompt template. Use {context} as placeholder.",
    )

    @field_validator("game", mode="before")
    @classmethod
    def ensure_game_defaults(cls, v):
        """Apply sensible defaults to GameConfig fields that may be missing."""
        if isinstance(v, dict):
            v.setdefault("type", "renpy")
            v.setdefault("path", ".")
        return v

    @classmethod
    def from_toml(cls, path: str | Path) -> Settings:
        """Load settings from a TOML file, then apply env var overrides."""
        # Python 3.11+ has tomllib built in; fall back to tomli for 3.10
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as fh:
            raw = tomllib.load(fh)

        return cls(**raw)

    @classmethod
    def from_env_only(cls) -> Settings:
        """Create settings purely from environment variables (no TOML file)."""
        return cls()  # Uses env prefix TESTER_ to pick up vars

    def effective_api_key(self) -> str | None:
        """Return the API key with fallback chain: config → env var → None."""
        if self.llm.api_key:
            return self.llm.api_key
        return os.environ.get("OPENAI_API_KEY", None)

    def config_dump(self) -> dict:
        """Return a sanitised dict for logging (redacts key)."""
        d = self.model_dump()
        if d.get("llm", {}).get("api_key"):
            d["llm"]["api_key"] = "***redacted***"
        return d