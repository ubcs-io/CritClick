"""Tests for launcher command construction and factory."""


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
