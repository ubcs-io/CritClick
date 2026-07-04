"""Pytest configuration and shared fixtures."""

import sys
from unittest.mock import patch

import pytest

from tester.config import HarnessSettings, LLMSettings, LoggingSettings, Settings
from tester.models import GameConfig


@pytest.fixture(autouse=True)
def _suppress_accessibility_check():
    """Suppress the macOS accessibility check during all Capturer inits.

    Without this, every ``Capturer()`` calls ``_check_accessibility()`` which
    tries to reach real Quartz/ApplicationServices APIs that may not be
    available or permitted in CI/dev environments.
    """
    with patch.object(sys, "platform", "linux"):
        yield


@pytest.fixture
def sample_game_config() -> GameConfig:
    return GameConfig(type="renpy", path="/fake/game", executable="renpy.sh")


@pytest.fixture
def sample_settings(sample_game_config: GameConfig) -> Settings:
    return Settings(
        llm=LLMSettings(api_base="http://localhost:8000/v1", model="test-model"),
        game=sample_game_config,
        harness=HarnessSettings(max_steps=5, startup_countdown=False),
        logging=LoggingSettings(save_screenshots=False),
    )
