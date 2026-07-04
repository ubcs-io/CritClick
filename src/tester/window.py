"""Cross-platform game-window detection and focus management.

Provides ``WindowTracker`` — find the game window by process PID,
activate (focus) it, and query its screen position / size so the
harness can crop captures and offset coordinates.

Platform support:
  - macOS: Quartz CoreGraphics (CGWindowListCopyWindowInfo)
  - Linux: xdotool (shell-out)
  - Windows: pygetwindow
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("tester.window")

# Timeout for polling until the game window appears
DEFAULT_WINDOW_FIND_TIMEOUT = 15.0
# Interval between poll attempts
_POLL_INTERVAL = 0.5


@dataclass(frozen=True)
class WindowInfo:
    """Screen position and size of the game window."""

    x: int
    y: int
    width: int
    height: int
    window_id: Any = None  # Platform-specific window handle


class WindowTracker:
    """Find and manage the game window on the local desktop.

    Usage::

        tracker = WindowTracker(pid=1234, timeout=15.0)
        win = tracker.find_window()
        if win:
            tracker.focus(win)
            # Crop capture to (win.x, win.y, win.width, win.height)
            # Add (win.x, win.y) to game-relative click coords
    """

    def __init__(
        self,
        pid: int,
        timeout: float = DEFAULT_WINDOW_FIND_TIMEOUT,
        debug: bool = False,
    ) -> None:
        self._pid = pid
        self._timeout = timeout
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_window(self) -> WindowInfo | None:
        """Poll until the game window appears or *timeout* elapses.

        Returns ``None`` if no window can be found (headless / Xvfb).
        """
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            win = self._find_platform_window()
            if win is not None:
                logger.info(
                    "🪟 Game window found: %dx%d at (%d, %d)",
                    win.width, win.height, win.x, win.y,
                )
                return win
            time.sleep(_POLL_INTERVAL)

        logger.warning(
            "⚠️  Could not find game window for PID %d after %.1fs. "
            "Falling back to full-screen capture + absolute coords.",
            self._pid, self._timeout,
        )
        return None

    @staticmethod
    def focus(win: WindowInfo) -> bool:
        """Bring *win* to the foreground so it receives input.

        Returns ``True`` on success, ``False`` if focus could not be
        verified (silently tolerated — clicks may still work).
        """
        system = platform.system()
        try:
            if system == "Darwin":
                return WindowTracker._focus_macos(win)
            elif system == "Linux":
                return WindowTracker._focus_linux(win)
            elif system == "Windows":
                return WindowTracker._focus_windows(win)
        except Exception as exc:
            logger.debug("🪟 Focus attempt failed (%s): %s", system, exc)
        return False

    # ------------------------------------------------------------------
    # Platform implementations — find
    # ------------------------------------------------------------------

    def _find_platform_window(self) -> WindowInfo | None:
        system = platform.system()
        try:
            if system == "Darwin":
                return self._find_macos()
            elif system == "Linux":
                return self._find_linux()
            elif system == "Windows":
                return self._find_windows()
        except Exception as exc:
            logger.debug("🪟 Window find failed (%s): %s", system, exc)
        return None

    # -- macOS ----------------------------------------------------------

    def _find_macos(self) -> WindowInfo | None:
        """Use Quartz CGWindowListCopyWindowInfo to find the game window by PID.

        Only returns on-screen windows with ``kCGWindowLayer == 0``
        (normal desktop windows, not the dock or menu bar).
        """
        try:
            import Quartz  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("🪟 Quartz (pyobjc-framework-Quartz) not available.")
            return None

        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        for entry in window_list:
            if entry.get(Quartz.kCGWindowOwnerPID) != self._pid:
                continue
            # kCGWindowLayer == 0 → normal desktop window
            if entry.get(Quartz.kCGWindowLayer, 99) != 0:
                continue

            bounds = entry.get(Quartz.kCGWindowBounds, {})
            x = int(bounds.get("X", 0))
            y = int(bounds.get("Y", 0))
            w = int(bounds.get("Width", 0))
            h = int(bounds.get("Height", 0))
            if w <= 0 or h <= 0:
                continue

            window_id = entry.get(Quartz.kCGWindowNumber)
            if self.debug:
                window_name = entry.get(Quartz.kCGWindowName, "")
                owner_name = entry.get(Quartz.kCGWindowOwnerName, "")
                window_layer = entry.get(Quartz.kCGWindowLayer, -1)
                window_alpha = entry.get(Quartz.kCGWindowAlpha, -1)
                logger.info(
                    "🔍 [DEBUG-WINDOW] macOS window: PID=%d | ID=%s | name='%s' | owner='%s' | "
                    "layer=%d alpha=%.2f | bounds=%dx%d+%d+%d",
                    self._pid, window_id, window_name, owner_name,
                    window_layer, window_alpha, w, h, x, y,
                )
            return WindowInfo(x=x, y=y, width=w, height=h, window_id=window_id)

        return None

    # -- Linux ----------------------------------------------------------

    def _find_linux(self) -> WindowInfo | None:
        """Use ``xdotool search --pid`` to find the game window."""
        if not self._has_xdotool():
            logger.debug("🪟 xdotool not available — cannot find game window.")
            return None

        try:
            result = subprocess.run(
                ["xdotool", "search", "--pid", str(self._pid)],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        window_ids = result.stdout.strip().splitlines()
        if not window_ids:
            return None

        # Take the first visible window for this PID
        wid = window_ids[0].strip()
        geo = self._xdotool_get_geometry(wid)
        if geo is None:
            return None

        x, y, w, h = geo
        if self.debug:
            logger.info(
                "🔍 [DEBUG-WINDOW] Linux window: PID=%d | ID=%s | %dx%d+%d+%d",
                self._pid, wid, w, h, x, y,
            )
        return WindowInfo(x=x, y=y, width=w, height=h, window_id=wid)

    # -- Windows --------------------------------------------------------

    def _find_windows(self) -> WindowInfo | None:
        """Use ``pygetwindow`` to find the game window by PID."""
        try:
            import pygetwindow as gw  # type: ignore[import-untyped]
            import win32process  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("🪟 pygetwindow / win32process not available.")
            return None

        for win in gw.getAllWindows():
            try:
                _, found_pid = win32process.GetWindowThreadProcessId(win._hWnd)
            except Exception:
                continue
            if found_pid != self._pid:
                continue
            if not win.visible or win.width <= 0 or win.height <= 0:
                continue
            if self.debug:
                logger.info(
                    "🔍 [DEBUG-WINDOW] Windows window: PID=%d | HWND=%s | %dx%d+%d+%d",
                    self._pid, win._hWnd, win.width, win.height, win.left, win.top,
                )
            return WindowInfo(
                x=win.left, y=win.top,
                width=win.width, height=win.height,
                window_id=win._hWnd,
            )

        return None

    # ------------------------------------------------------------------
    # Platform implementations — focus
    # ------------------------------------------------------------------

    @staticmethod
    def _focus_macos(win: WindowInfo) -> bool:
        """Bring window to front using ``osascript`` / AppleScript.

        Falls back to Quartz if AppleScript is unavailable.
        """
        # Preferred: AppleScript (reliable, no extra deps)
        try:
            script = '''
            tell application "System Events"
                set frontmost of process whose unix id is {pid} to true
            end tell
            '''.replace("{pid}", str(win.window_id or 0))
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                logger.debug("🪟 Focused macOS window via AppleScript.")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: Quartz
        try:
            import Quartz  # type: ignore[import-untyped]

            if win.window_id is not None:
                Quartz.CGWindowListCreateDescriptionFromArray([win.window_id])
                # No direct focus API in Quartz — AppleScript is better
        except ImportError:
            pass

        return False

    @staticmethod
    def _focus_linux(win: WindowInfo) -> bool:
        """Focus using ``xdotool windowactivate``."""
        if win.window_id is None:
            return False
        try:
            result = subprocess.run(
                ["xdotool", "windowactivate", str(win.window_id)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                logger.debug("🪟 Focused Linux window via xdotool.")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    @staticmethod
    def _focus_windows(win: WindowInfo) -> bool:
        """Focus using ``pygetwindow``."""
        try:
            import pygetwindow as gw  # type: ignore[import-untyped]

            for w in gw.getAllWindows():
                if w._hWnd == win.window_id:
                    w.activate()
                    logger.debug("🪟 Focused Windows window via pygetwindow.")
                    return True
        except ImportError:
            pass
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_xdotool() -> bool:
        """Check whether ``xdotool`` is available on PATH."""
        try:
            result = subprocess.run(
                ["which", "xdotool"],
                capture_output=True, text=True, timeout=3,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _xdotool_get_geometry(window_id: str) -> tuple[int, int, int, int] | None:
        """Return (x, y, width, height) from ``xdotool getwindowgeometry``."""
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", window_id],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0:
            return None

        # Parse output like:
        #   Window 12345678
        #     Position: 100,200 (screen: 0)
        #     Geometry: 1280x720
        import re

        pos_match = re.search(r"Position:\s*(\d+),(\d+)", result.stdout)
        geo_match = re.search(r"Geometry:\s*(\d+)x(\d+)", result.stdout)

        if not pos_match or not geo_match:
            return None

        return (
            int(pos_match.group(1)),
            int(pos_match.group(2)),
            int(geo_match.group(1)),
            int(geo_match.group(2)),
        )