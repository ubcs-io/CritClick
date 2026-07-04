"""Game process launchers for Ren'Py, Godot, and custom executables.

Each launcher handles:
  - Finding the right executable in the game directory
  - Launching the game with appropriate CLI flags
  - Lifecycle management (start, stop, health check)
  - Optional git pull before launch
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .models import GameConfig

logger = logging.getLogger("tester.launcher")


def capture_git_info(path: str | os.PathLike) -> dict:
    """Return ``{commit, branch, dirty}`` for the git repo containing *path*.

    Gracefully handles missing git, non-repo directories, and command failures
    by returning ``None`` for any field that can't be determined. Designed for
    inclusion in a run manifest so playthroughs can be correlated to code state.
    """
    info: dict[str, str | bool | None] = {"commit": None, "branch": None, "dirty": None}
    repo = Path(path)
    if repo.is_file():
        repo = repo.parent

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    # Walk up until we find a .git dir (rev-parse handles this centrally)
    rev_parse = _git("rev-parse", "--show-toplevel")
    if rev_parse is None:
        return info
    repo = Path(rev_parse)

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    info["commit"] = commit
    info["branch"] = branch
    info["dirty"] = bool(status)
    return info


class Launcher(ABC):
    """Abstract base for game launchers."""

    def __init__(self, config: GameConfig, headless: bool = True) -> None:
        self.config = config
        # ``headless`` selects the Linux virtual-display (Xvfb) code path.
        # On macOS/Windows dev machines the game renders to the real desktop,
        # so callers should pass ``headless=False`` (or set [harness].headless).
        self.headless = headless
        self.process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def resolve_executable(self) -> str:
        """Return the full path to the game executable."""
        ...

    def launch(self) -> None:
        """Start the game process."""
        if self.process is not None:
            logger.warning("Game is already running. Skipping launch.")
            return

        exe = self.resolve_executable()
        cmd = self.build_command(exe)
        logger.info(f"🚀 Launching: {' '.join(shlex.quote(p) for p in cmd)}")

        env = os.environ.copy()
        # Only touch DISPLAY on Linux headless runs (Xvfb). On macOS/Windows
        # the game renders to the interactive desktop and DISPLAY is ignored.
        if sys.platform == "linux" and self.headless and self.config.resolution:
            env.setdefault("DISPLAY", ":0")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        # Give the game time to initialise
        time.sleep(5)

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the game process gracefully, then forcefully if needed."""
        if self.process is None:
            return
        logger.info("🛑 Stopping game…")
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Game did not terminate gracefully — killing.")
            self.process.kill()
            self.process.wait()
        self.process = None

    def is_alive(self) -> bool:
        """Check whether the game process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_pid(self) -> int | None:
        """Return the PID of the launched game process, or ``None`` if not running."""
        if self.process is None:
            return None
        return self.process.pid

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    @abstractmethod
    def build_command(self, executable: str) -> list[str]:
        """Build the full CLI command list from the game config."""
        ...

    # ------------------------------------------------------------------
    # Git pull
    # ------------------------------------------------------------------

    def pull_latest(self) -> None:
        """If configured, run ``git pull --rebase`` in the game directory."""
        if not self.config.git_pull:
            return
        game_dir = Path(self.config.path)
        git_dir = game_dir / ".git"
        if not git_dir.is_dir():
            logger.info("Not a git repository — skipping pull.")
            return
        logger.info("📥 Pulling latest version from git…")
        try:
            result = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=str(game_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                logger.info("✅ Git pull successful.")
            else:
                logger.warning(f"⚠️  Git pull failed:\n{result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning(f"⚠️  Git pull error: {exc}")


# ---------------------------------------------------------------------------
# Ren'Py launcher
# ---------------------------------------------------------------------------

class RenPyLauncher(Launcher):
    """Launches a Ren'Py game (renpy.sh on Linux/macOS, renpy.exe on Windows)."""

    RENPY_EXECUTABLES = {
        "darwin": "renpy.sh",
        "linux": "renpy.sh",
        "win32": "renpy.exe",
    }

    def resolve_executable(self) -> str:
        explicit = self.config.executable
        if explicit:
            exe = shutil.which(explicit) or explicit
            if os.path.isfile(exe):
                return exe
            # Try relative to game path
            alt = os.path.join(self.config.path, explicit)
            if os.path.isfile(alt):
                return alt

        # Auto-detect based on platform
        exe_name = self.RENPY_EXECUTABLES.get(sys.platform, "renpy.sh")
        candidates = [
            os.path.join(self.config.path, exe_name),
            os.path.join(self.config.path, "renpy", exe_name),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"Ren'Py executable not found in '{self.config.path}'. "
            f"Searched: {candidates}. Set [game].executable in your config."
        )

    def build_command(self, executable: str) -> list[str]:
        cmd = [executable, self.config.path]
        # Ren'Py does not support a --size CLI flag; resolution is controlled
        # inside the game itself (options.rpy / gui.rpy). The GameConfig
        # resolution field is used by Godot and for coordinate validation but
        # has no effect on the Ren'Py command line.
        cmd += self.config.args
        return cmd


# ---------------------------------------------------------------------------
# Godot launcher
# ---------------------------------------------------------------------------

class GodotLauncher(Launcher):
    """Launches a Godot game from a project file or exported binary.

    For development (project.godot), requires the Godot editor binary.
    For exported builds, point to the exported executable directly via
    [game].executable.
    """

    def resolve_executable(self) -> str:
        explicit = self.config.executable
        if explicit:
            exe = shutil.which(explicit) or explicit
            if os.path.isfile(exe):
                return exe
        # Default: look for 'godot' on PATH
        exe = shutil.which("godot")
        if exe:
            return exe
        raise FileNotFoundError(
            "Godot executable not found. Set [game].executable in your config, "
            "or ensure 'godot' is on your PATH."
        )

    def build_command(self, executable: str) -> list[str]:
        game_path = Path(self.config.path)
        cmd = [executable]

        if game_path.is_dir() and (game_path / "project.godot").is_file():
            # Development mode — open project
            cmd += ["--path", str(game_path)]
        elif game_path.suffix == ".godot":
            # Pointing directly at a project file
            cmd += ["--path", str(game_path.parent)]
        else:
            # Assume it's an exported binary or main scene
            cmd.append(str(game_path))

        w, h = self.config.resolution or (1280, 720)
        cmd += ["--resolution", f"{w}x{h}", "--fullscreen"]
        # `--no-window` is a headless-Linux concern. On a dev machine we want
        # the window visible so the user can watch the playthrough.
        if self.headless:
            cmd.append("--no-window")  # may conflict; user can override via args
        cmd += self.config.args

        return cmd


# ---------------------------------------------------------------------------
# Custom launcher
# ---------------------------------------------------------------------------

class CustomLauncher(Launcher):
    """Launches any executable with optional arguments."""

    def resolve_executable(self) -> str:
        explicit = self.config.executable or self.config.path
        exe = shutil.which(explicit) or explicit
        if not os.path.isfile(exe):
            raise FileNotFoundError(f"Custom executable not found: {exe}")
        return exe

    def build_command(self, executable: str) -> list[str]:
        return [executable, *self.config.args]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LAUNCHER_MAP = {
    "renpy": RenPyLauncher,
    "godot": GodotLauncher,
    "custom": CustomLauncher,
}


def create_launcher(config: GameConfig, headless: bool = True) -> Launcher:
    """Factory: return the appropriate Launcher for the given game config.

    Args:
        config: Game configuration.
        headless: When True (default, for Linux servers), use the Xvfb/headless
            code path. Pass False for interactive macOS/Windows dev runs so the
            game window is visible and DISPLAY is left untouched.
    """
    cls = _LAUNCHER_MAP.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unsupported game type: '{config.type}'. "
            f"Supported types: {list(_LAUNCHER_MAP)}"
        )
    return cls(config, headless=headless)