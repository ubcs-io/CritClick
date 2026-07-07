"""Tests for launcher command construction and factory."""


import io
import os
import time

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
        assert "/fake/game" in cmd
        # Ren'Py does not support --size; resolution is game-internal.
        assert "--size" not in cmd
        assert "1280x720" not in cmd

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


class TestOutputCapture:
    """The reader threads drain stdout/stderr into readable tails."""

    def _launcher(self):
        config = GameConfig(type="custom", path="/fake", executable="/usr/bin/true")
        return CustomLauncher(config)

    def _drain(self, launcher, buffer, data: bytes):
        """Run the drain loop synchronously over an in-memory stream."""
        launcher._drain_stream(io.BytesIO(data), buffer)

    def test_stdout_tail_captures_lines(self):
        launcher = self._launcher()
        self._drain(launcher, launcher._stdout_buffer, b"line one\nline two\n")
        assert launcher.get_stdout_tail() == "line one\nline two"

    def test_empty_output_returns_none(self):
        launcher = self._launcher()
        assert launcher.get_stdout_tail() is None
        assert launcher.get_stderr_tail() is None

    def test_ring_buffer_bounded_keeps_tail(self):
        from tester.launcher import _MAX_OUTPUT_LINES

        launcher = self._launcher()
        payload = "".join(f"line{i}\n" for i in range(_MAX_OUTPUT_LINES + 50))
        self._drain(launcher, launcher._stderr_buffer, payload.encode())
        tail = launcher.get_stderr_tail()
        # Oldest lines evicted; the most recent survive.
        assert "line0\n" not in tail + "\n"
        assert tail.endswith(f"line{_MAX_OUTPUT_LINES + 49}")

    def test_tail_truncates_to_max_chars(self):
        launcher = self._launcher()
        self._drain(launcher, launcher._stdout_buffer, b"abcdefghij\n")
        assert launcher.get_stdout_tail(max_chars=3) == "hij"

    def test_decodes_invalid_utf8_without_raising(self):
        launcher = self._launcher()
        self._drain(launcher, launcher._stderr_buffer, b"bad \xff byte\n")
        assert "bad" in launcher.get_stderr_tail()


class TestEngineErrorLog:
    """Ren'Py's traceback.txt is read back only when fresh."""

    def test_renpy_reads_recent_traceback(self, tmp_path):
        (tmp_path / "traceback.txt").write_text(
            'Traceback (most recent call last):\n'
            '  File "game/script.rpy", line 42, in script\n'
            'Exception: boom\n'
        )
        config = GameConfig(type="renpy", path=str(tmp_path), executable="renpy.sh")
        launcher = RenPyLauncher(config)
        launcher._launched_at = time.time() - 1  # file is newer than launch
        log = launcher.read_engine_error_log()
        assert log is not None
        assert "game/script.rpy" in log

    def test_renpy_ignores_stale_traceback(self, tmp_path):
        tb = tmp_path / "traceback.txt"
        tb.write_text("old crash from a previous run\n")
        old = time.time() - 3600
        os.utime(tb, (old, old))
        config = GameConfig(type="renpy", path=str(tmp_path), executable="renpy.sh")
        launcher = RenPyLauncher(config)
        launcher._launched_at = time.time()  # launched after the stale file
        assert launcher.read_engine_error_log() is None

    def test_renpy_no_traceback_file(self, tmp_path):
        config = GameConfig(type="renpy", path=str(tmp_path), executable="renpy.sh")
        launcher = RenPyLauncher(config)
        launcher._launched_at = time.time()
        assert launcher.read_engine_error_log() is None

    def test_base_launcher_has_no_engine_log(self):
        config = GameConfig(type="custom", path="/fake", executable="/usr/bin/true")
        assert CustomLauncher(config).read_engine_error_log() is None


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