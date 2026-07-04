"""Tests for configuration loading and validation."""

import argparse
from pathlib import Path

import pytest

from tester.config import (
    HarnessSettings,
    LLMSettings,
    Settings,
)
from tester.models import GameConfig


class TestLLMSettings:
    def test_defaults(self):
        s = LLMSettings()
        assert s.api_base == "https://api.openai.com/v1"
        assert s.model == "gpt-4o"
        assert s.max_retries == 3
        assert s.retry_delay == 2.0

    def test_custom_values(self):
        s = LLMSettings(
            api_base="http://localhost:11434/v1",
            model="llava",
            max_retries=5,
            retry_delay=0.5,
        )
        assert s.api_base == "http://localhost:11434/v1"
        assert s.max_retries == 5
        assert s.retry_delay == 0.5

    def test_block_external_routing_default(self):
        s = LLMSettings()
        assert s.block_external_routing is False

    def test_block_external_routing_localhost_accepted(self):
        s = LLMSettings(
            api_base="http://localhost:11434/v1",
            block_external_routing=True,
        )
        assert s.block_external_routing is True

    def test_block_external_routing_loopback_ip_accepted(self):
        s = LLMSettings(
            api_base="http://127.0.0.1:8000/v1",
            block_external_routing=True,
        )
        assert s.block_external_routing is True

    def test_block_external_routing_public_ip_rejected(self):
        with pytest.raises(Exception):
            LLMSettings(
                api_base="https://api.openai.com/v1",
                block_external_routing=True,
            )

    def test_block_external_routing_private_ip_accepted(self):
        s = LLMSettings(
            api_base="http://192.168.1.100:11434/v1",
            block_external_routing=True,
        )
        assert s.block_external_routing is True

    def test_block_external_routing_disabled_allows_public(self):
        s = LLMSettings(
            api_base="https://api.openai.com/v1",
            block_external_routing=False,
        )
        assert s.block_external_routing is False


class TestHarnessSettings:
    def test_defaults(self):
        s = HarnessSettings()
        assert s.max_steps == 50
        assert s.wait_after_action == 1.5
        assert s.stuck_threshold == 3
        assert s.screen_scale == 1.0
        assert s.wait_duration == 2.0
        assert s.startup_countdown is True

    def test_custom_values(self):
        s = HarnessSettings(
            max_steps=10,
            wait_duration=5.0,
            startup_countdown=False,
        )
        assert s.max_steps == 10
        assert s.wait_duration == 5.0
        assert s.startup_countdown is False


class TestSettingsFromTOML:
    def test_load_valid_toml(self, tmp_path: Path):
        toml_content = """
[llm]
api_base = "http://localhost:8000/v1"
model = "test-model"
max_retries = 5
retry_delay = 1.0

[game]
type = "renpy"
path = "/fake/game"
executable = "renpy.sh"

[harness]
max_steps = 10
wait_duration = 3.0
startup_countdown = false

[logging]
save_screenshots = false
"""
        toml_file = tmp_path / "test_config.toml"
        toml_file.write_text(toml_content)

        settings = Settings.from_toml(toml_file)
        assert settings.llm.api_base == "http://localhost:8000/v1"
        assert settings.llm.model == "test-model"
        assert settings.llm.max_retries == 5
        assert settings.harness.max_steps == 10
        assert settings.harness.wait_duration == 3.0
        assert settings.harness.startup_countdown is False

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            Settings.from_toml("/nonexistent/path.toml")


class TestEffectiveAPIKey:
    def test_from_config(self):
        settings = Settings(
            llm=LLMSettings(api_key="config-key"),
            game=GameConfig(type="renpy", path="."),
        )
        assert settings.effective_api_key() == "config-key"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        settings = Settings(
            llm=LLMSettings(),
            game=GameConfig(type="renpy", path="."),
        )
        assert settings.effective_api_key() == "env-key"

    def test_config_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        settings = Settings(
            llm=LLMSettings(api_key="config-key"),
            game=GameConfig(type="renpy", path="."),
        )
        assert settings.effective_api_key() == "config-key"

    def test_none_when_not_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(
            llm=LLMSettings(),
            game=GameConfig(type="renpy", path="."),
        )
        assert settings.effective_api_key() is None


class TestConfigDump:
    def test_redacts_api_key(self):
        settings = Settings(
            llm=LLMSettings(api_key="secret-key-12345"),
            game=GameConfig(type="renpy", path="."),
        )
        d = settings.config_dump()
        assert d["llm"]["api_key"] == "***redacted***"
        assert "secret-key-12345" not in str(d)


# ---------------------------------------------------------------------------
# Screen size override tests (apply_screen_preset from cli.py)
# ---------------------------------------------------------------------------


class TestApplyScreenPreset:
    """Tests for the ``apply_screen_preset`` helper that mutates ``GameConfig.resolution``."""

    @staticmethod
    def _make_args(mobile=False, tablet=False, landscape=False):
        """Build an argparse.Namespace with the screen preset flags."""
        return argparse.Namespace(mobile=mobile, tablet=tablet, landscape=landscape)

    @staticmethod
    def _make_game_config():
        return GameConfig(type="renpy", path="/fake/game")

    def test_mobile_portrait(self):
        from tester.cli import apply_screen_preset

        args = self._make_args(mobile=True)
        gc = self._make_game_config()
        apply_screen_preset(args, gc)
        assert gc.resolution == (390, 844)

    def test_mobile_landscape(self):
        from tester.cli import apply_screen_preset

        args = self._make_args(mobile=True, landscape=True)
        gc = self._make_game_config()
        apply_screen_preset(args, gc)
        assert gc.resolution == (844, 390)

    def test_tablet_portrait(self):
        from tester.cli import apply_screen_preset

        args = self._make_args(tablet=True)
        gc = self._make_game_config()
        apply_screen_preset(args, gc)
        assert gc.resolution == (1024, 768)

    def test_tablet_landscape(self):
        from tester.cli import apply_screen_preset

        args = self._make_args(tablet=True, landscape=True)
        gc = self._make_game_config()
        apply_screen_preset(args, gc)
        assert gc.resolution == (768, 1024)

    def test_mobile_and_tablet_mutually_exclusive(self, capsys):
        from tester.cli import apply_screen_preset

        args = self._make_args(mobile=True, tablet=True)
        gc = self._make_game_config()
        with pytest.raises(SystemExit):
            apply_screen_preset(args, gc)
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err

    def test_landscape_without_preset_is_noop(self, capsys):
        from tester.cli import apply_screen_preset

        args = self._make_args(landscape=True)
        gc = self._make_game_config()
        original = gc.resolution
        apply_screen_preset(args, gc)
        assert gc.resolution == original
        captured = capsys.readouterr()
        assert "no effect" in captured.err

    def test_no_flags_leaves_resolution_unchanged(self):
        from tester.cli import apply_screen_preset

        args = self._make_args()
        gc = self._make_game_config()
        original = gc.resolution
        apply_screen_preset(args, gc)
        assert gc.resolution == original
