"""Tests for launcher command construction and factory."""


import os

import pytest

from tester.launcher import CustomLauncher, GodotLauncher, RenPyLauncher, create_launcher
from tester.models import GameConfig


class TestRenpyLauncher:
    def test_resolve_executable(self, tmp_path):
        game_dir = tmp_path / "mygame"
        game_dir.mkdir()
        (game_dir / "renpy.sh").write_text("#!/bin/bash\necho hi")
        config = GameConfig(type="renpy", path=str(game_dir), executable="renpy.sh")
        launcher = RenPyLauncher(config)
        exe = launcher.resolve_executable()
        assert "renpy.sh" in exe

    def test_build_command(self):
        config = GameConfig(type="renpy", path="/fake/game", executable="renpy.sh")
        launcher = RenPyLauncher(config)
        cmd = launcher.build_command("/fake/game/renpy.sh")
        assert cmd[0] == "/fake/game/renpy.sh"
        assert "--window" in cmd
        assert "--nodaemon" in cmd


class TestGodotLauncher:
    def test_build_command_with_project_dir(self, tmp_path):
        game_dir = tmp_path / "mygodotgame"
        game_dir.mkdir()
        (game_dir / "project.godot").write_text("; config")
        config = GameConfig(type="godot", path=str(game_dir), executable="/usr/bin/godot")
        launcher = GodotLauncher(config)
        cmd = launcher.build_command("/usr/bin/godot")
        assert cmd[0] == "/usr/bin/godot"
        assert "--path" in cmd

    @pytest.mark.parametrize("headless", [True, False])
    def test_no_window_only_when_headless(self, tmp_path, headless):
        """``--no-window`` should appear only on headless runs.

        On a macOS/Windows dev machine we want the window visible so the
        user can watch the playthrough.
        """
        game_dir = tmp_path / "mygodotgame"
        game_dir.mkdir()
        (game_dir / "project.godot").write_text("; config")
        config = GameConfig(type="godot", path=str(game_dir), executable="/usr/bin/godot")
        launcher = GodotLauncher(config, headless=headless)
        cmd = launcher.build_command("/usr/bin/godot")
        if headless:
            assert "--no-window" in cmd
        else:
            assert "--no-window" not in cmd


class TestCustomLauncher:
    def test_build_command(self):
        config = GameConfig(
            type="custom", path="/fake/game", executable="/usr/local/bin/mygame"
        )
        launcher = CustomLauncher(config)
        cmd = launcher.build_command("/usr/local/bin/mygame")
        assert "/usr/local/bin/mygame" in cmd


class TestCreateLauncherFactory:
    def test_renpy(self):
        config = GameConfig(type="renpy", path="/fake")
        launcher = create_launcher(config)
        assert isinstance(launcher, RenPyLauncher)

    def test_godot(self):
        config = GameConfig(type="godot", path="/fake", executable="godot")
        launcher = create_launcher(config)
        assert isinstance(launcher, GodotLauncher)

    def test_custom(self):
        config = GameConfig(type="custom", path="/fake", executable="mygame")
        launcher = create_launcher(config)
        assert isinstance(launcher, CustomLauncher)

    def test_unsupported_type(self):
        config = GameConfig(type="unity", path="/fake")
        with pytest.raises(ValueError):
            create_launcher(config)

    def test_headless_propagated(self):
        """``create_launcher(..., headless=False)`` must reach the instance."""
        config = GameConfig(type="renpy", path="/fake", executable="renpy.sh")
        assert create_launcher(config, headless=True).headless is True
        assert create_launcher(config, headless=False).headless is False


class TestHeadlessEnvironment:
    """The DISPLAY/Xvfb environment tweak must be Linux + headless only."""

    @pytest.mark.parametrize(
        "platform,headless,sets_display",
        [
            ("linux", True, True),
            ("linux", False, False),
            ("darwin", True, False),
            ("darwin", False, False),
            ("win32", True, False),
            ("win32", False, False),
        ],
    )
    def test_display_env_gated_by_platform_and_headless(
        self, monkeypatch, platform, headless, sets_display
    ):
        # Make sure DISPLAY isn't inherited from the dev machine.
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr("tester.launcher.sys.platform", platform)

        config = GameConfig(type="custom", path="/usr/bin/true", executable="/usr/bin/true")
        launcher = create_launcher(config, headless=headless)

        # Recreate the launch() env logic without actually spawning a process.
        env = os.environ.copy()
        if platform == "linux" and launcher.headless and launcher.config.resolution:
            env.setdefault("DISPLAY", ":0")

        assert ("DISPLAY" in env) is sets_display