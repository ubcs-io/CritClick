"""Configuration loading from TOML files with environment variable overrides.

Resolution order (last wins):
  1. Defaults hardcoded in the Settings model
  2. Values loaded from a TOML config file
  3. Environment variables (e.g. TESTER_API_KEY, TESTER_API_BASE, …)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import GameConfig


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
    api_key: Optional[str] = Field(
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
    max_tokens: int = Field(default=600, ge=1, le=16384)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


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
    headless: bool = Field(
        default=False,
        description="If True, attempt to run on a virtual display (Linux + Xvfb).",
    )


# ---------------------------------------------------------------------------
# Logging / output settings
# ---------------------------------------------------------------------------

class LoggingSettings(BaseModel):
    """Controls what and how the harness records during a playthrough."""

    log_file: str = Field(default="playthrough_log.jsonl")
    save_screenshots: bool = Field(default=True)
    screenshot_dir: str = Field(default="screenshots")
    context_window_size: int = Field(
        default=3, ge=0,
        description="Number of recent narrative entries to include in the LLM prompt.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


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

    # Prompt overrides (optional — use defaults if omitted)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Override the default system prompt for the LLM.",
    )
    user_prompt_template: Optional[str] = Field(
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
        import tomli

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as fh:
            raw = tomli.load(fh)

        return cls(**raw)

    @classmethod
    def from_env_only(cls) -> Settings:
        """Create settings purely from environment variables (no TOML file)."""
        return cls()  # Uses env prefix TESTER_ to pick up vars

    def effective_api_key(self) -> Optional[str]:
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
