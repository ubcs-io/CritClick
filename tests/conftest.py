"""Pytest configuration and shared fixtures."""


import pytest

from tester.config import HarnessSettings, LLMSettings, LoggingSettings, Settings
from tester.models import GameConfig


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
