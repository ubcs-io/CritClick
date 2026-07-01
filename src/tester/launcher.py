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
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import GameConfig

logger = logging.getLogger("tester.launcher")


class Launcher(ABC):
    """Abstract base for game launchers."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None

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

        # On Linux with headless mode, set up DISPLAY
        env = os.environ.copy()
        if self.config.resolution:
            env.setdefault("DISPLAY", ":0")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        w, h = self.config.resolution or (1280, 720)
        cmd = [
            executable,
            "--window",
            "--size", f"{w}x{h}",
            "--nodaemon",
            *self.config.args,
        ]
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
        cmd += ["--resolution", f"{w}x{h}", "--fullscreen", "--no-window"]  # may conflict; user can override via args
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


def create_launcher(config: GameConfig) -> Launcher:
    """Factory: return the appropriate Launcher for the given game config."""
    cls = _LAUNCHER_MAP.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unsupported game type: '{config.type}'. "
            f"Supported types: {list(_LAUNCHER_MAP)}"
        )
    return cls(config)
