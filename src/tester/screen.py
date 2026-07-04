"""Screen capture utilities for the vision harness.

Uses ``mss`` (fast, cross-platform) as the primary capture method, with
``pyautogui`` as a cross-platform fallback. On Linux, ``mss`` uses X11/shm
and works both on the local display and inside Xvfb virtual framebuffers.

When a game-window position is provided (via ``set_game_window``),
captures are cropped to the game window and click coordinates are offset
automatically so the LLM can work in game-window-relative space.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image

from .window import WindowInfo

if TYPE_CHECKING:
    from .window import WindowTracker

logger = logging.getLogger("tester.screen")


class ScreenCaptureError(RuntimeError):
    """Raised when screen capture fails."""


class AccessibilityError(RuntimeError):
    """Raised when the tool lacks macOS Accessibility permissions for mouse control."""


class Capturer:
    """Captures the screen (or a region) and returns PNG bytes / base64.

    Supports coordinate scaling for DPI / resolution mismatches.
    On macOS, auto-detects Retina backing-scale and applies the correction
    automatically so that LLM-returned pixel coordinates land on the
    correct logical position.

    When ``set_game_window()`` has been called, captures are cropped to
    the game window and click coordinates are offset by the window's
    screen position — the LLM receives only the game window and returns
    coordinates relative to it.
    """

    def __init__(
        self,
        scale: float = 0.0,
        debug: bool = False,
        game_resolution: tuple[int, int] | None = None,
    ) -> None:
        # ``scale`` acts as a manual override.  When zero (default), the
        # capturer will auto-detect the display scale factor where possible.
        self._manual_scale = scale if scale > 0 else None
        self._mss_available = False
        self._mss = None
        self.debug = debug
        if self.debug:
            logger.info("🔍 [DEBUG-SCREEN] Capturer initialised in debug mode.")

        # Try to import mss (fast cross-platform capture)
        try:
            import mss

            self._mss = mss.MSS()
            self._mss_available = True
            logger.debug("Using mss for screen capture.")
        except ImportError:
            logger.debug("mss not available — falling back to pyautogui.")

        # --- Coordinate scale -------------------------------------------------
        self._scale = self._resolve_scale()

        # --- Game window tracking ---------------------------------------------
        self._game_window: WindowInfo | None = None
        self._window_tracker: WindowTracker | None = None
        # Physical-pixel window position (accounting for Retina scale).
        # Populated by set_game_window() — used for cropping and offset.
        self._game_window_physical: WindowInfo | None = None

        # --- macOS accessibility check ----------------------------------------
        if sys.platform == "darwin":
            self._check_accessibility()

    # ------------------------------------------------------------------
    # Game window binding
    # ------------------------------------------------------------------

    def set_game_window(self, win: WindowInfo | None, tracker: WindowTracker | None = None) -> None:
        """Bind to a game window for cropped capture and offset clicks.

        Args:
            win: The game window info (position + size) in **logical** points
                (as returned by Quartz/pygetwindow).  On Retina/HiDPI displays
                these are NOT physical pixels.  This method scales the bounds
                to physical pixels for correct cropping of mss captures.

            tracker: Optional ``WindowTracker`` used to re-focus the
                window before each click.
        """
        self._game_window = win
        self._window_tracker = tracker
        if win is not None:
            logger.info(
                "🪟 Capturer bound to game window (logical): %dx%d at (%d, %d)",
                win.width, win.height, win.x, win.y,
            )

            # On Retina/HiDPI displays, the window bounds from Quartz are in
            # **logical points** while mss captures in **physical pixels**.
            # Multiply by the Retina scale to get physical-pixel coordinates
            # so that the crop and offset work correctly.
            #
            # The scale stored in self._scale is the logical→physical
            # multiplier (e.g. 0.5 on a 2× display means logical pixels are
            # half the size).  The physical→logical factor is 1/self._scale.
            retina = 1.0 / self._scale if self._scale > 0 else 1.0
            phys_x = int(win.x * retina)
            phys_y = int(win.y * retina)
            phys_w = int(win.width * retina)
            phys_h = int(win.height * retina)

            self._game_window_physical = WindowInfo(
                x=phys_x, y=phys_y, width=phys_w, height=phys_h,
                window_id=win.window_id,
            )

            logger.info(
                "🪟 Capturer bound (physical pixels): %dx%d at (%d, %d) "
                "[retina=%.1fx]",
                phys_w, phys_h, phys_x, phys_y, retina,
            )
        else:
            self._game_window_physical = None
            logger.debug("🪟 Capturer unbound from game window — full-screen mode.")

    @property
    def game_window(self) -> WindowInfo | None:
        """The currently bound game window, or ``None``."""
        return self._game_window

    @property
    def scale(self) -> float:
        """Coordinate scale factor (read-only).

        On Retina/HiDPI displays, captured screenshots are in physical
        pixels while pyautogui operates in logical points.  This scale
        maps LLM coordinates (in the physical-pixel image space) to
        logical points that pyautogui can use.

        When a game window is bound and the crop has been correctly
        scaled to physical pixels, the LLM sees the game at its native
        rendered resolution — no additional window→game scale needed.
        """
        return self._scale

    # ------------------------------------------------------------------
    # Scale detection
    # ------------------------------------------------------------------

    def _resolve_scale(self) -> float:
        """Determine the effective coordinate scale factor.

        Priority:
          1. Manual ``scale`` passed to ``__init__`` (if > 0).
          2. Auto-detected backing-scale factor on macOS.
          3. Logical-to-physical resolution ratio on other platforms.
          4. Fallback to ``1.0``.
        """
        if self._manual_scale is not None:
            logger.info("🧮 Using manual screen_scale=%.2f", self._manual_scale)
            return self._manual_scale

        # macOS: query the system for the display backing-scale factor
        if sys.platform == "darwin":
            detected = self._detect_macos_retina_scale()
            if detected is not None:
                # The LLM sees the physical-pixel screenshot, but pyautogui
                # works in logical points.  E.g. on a 2x display,
                # physical=3024 → logical=1512, so *multiply* LLM coords by
                # 1/scale (= 0.5) to map physical → logical.
                correction = 1.0 / detected
                logger.info(
                    "🖥️  Auto-detected macOS Retina scale %.1fx → coordinate multiplier %.2f",
                    detected,
                    correction,
                )
                return correction

        # Cross-platform heuristic: compare the *physical-pixel* capture
        # size (mss when available, otherwise pyautogui) against
        # pyautogui's reported logical screen size.  On a Retina /
        # HiDPI display the physical capture will be 2× (or 3×) the
        # logical dimensions, giving us the correct coordinate multiplier.
        try:
            import pyautogui

            logical_w, logical_h = pyautogui.size()
            # Use the actual full-screen capture path (mss if available)
            # so we get physical pixels, not logical points.
            phys_img = self._capture_full_screen()
            phys_w, phys_h = phys_img.size
            if phys_w > 0 and phys_h > 0 and logical_w > 0 and logical_h > 0:
                scale_w = logical_w / phys_w
                scale_h = logical_h / phys_h
                avg = (scale_w + scale_h) / 2.0
                if abs(avg - 1.0) > 0.05:
                    logger.info(
                        "🧮 Physical screenshot %dx%d → logical %dx%d → scale %.2f",
                        phys_w, phys_h, logical_w, logical_h, avg,
                    )
                    return avg
        except Exception:
            pass

        return 1.0

    @staticmethod
    def _detect_macos_retina_scale() -> float | None:
        """Return the backing-scale factor of the built-in Retina display.

        Uses ``system_profiler SPDisplaysDataType`` and parses the
        *Resolution* line (e.g. ``"2560 x 1600 Retina"``) against the
        *UI Looks like* line (e.g. ``"2560 x 1600 @ 2.00x"``).

        Returns ``None`` if detection fails (no Retina display, or the
        command isn't available).
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0:
            return None

        # Parse the output.  Scan for a pair of "Resolution:" and
        # "UI Looks like:" lines that belong to the same display.
        # Because the output may have multiple displays, we track the
        # most recently seen resolution and match it against the next
        # UI Looks like line.
        import re

        last_resolution_line: int | None = None

        for lineno, line in enumerate(result.stdout.splitlines()):
            stripped = line.strip()

            m = re.match(r"Resolution:\s*\d+\s*x\s*\d+", stripped, re.IGNORECASE)
            if m:
                last_resolution_line = lineno
                continue

            m = re.match(
                r"UI Looks like:\s*\d+\s*x\s*\d+\s*@\s*(\d+\.?\d*)x",
                stripped,
                re.IGNORECASE,
            )
            if m and last_resolution_line is not None:
                # Only accept if the UI Looks like line is within a few
                # lines of the Resolution line (same display block).
                if lineno - last_resolution_line <= 5:
                    return float(m.group(1))

        return None

    # ------------------------------------------------------------------
    # macOS Accessibility permission check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_accessibility() -> None:
        """Warn loudly if macOS Accessibility permission is missing.

        On macOS, pyautogui uses Quartz APIs that require the terminal
        (or the IDE) to be listed under *System Settings → Privacy &
        Security → Accessibility*.  Without it, ``pyautogui.click()``
        silently fails.
        """
        # Quickest check: try to move the mouse by 0 pixels.  If it raises
        # a permission denied error, we know accessibility isn't granted.
        try:
            import pyautogui

            # Save current position and try a 0-delta move
            pos = pyautogui.position()
            pyautogui.moveTo(pos.x, pos.y)
        except Exception:
            logger.warning(
                "⚠️  macOS Accessibility permission may be missing. "
                "Grant it in System Settings → Privacy & Security → Accessibility. "
                "Without it, mouse clicks will silently fail."
            )
            return

        # Fallback: use the macOS ApplicationServices AX API
        try:
            from ApplicationServices import (  # type: ignore[import-untyped]
                AXIsProcessTrusted,
            )

            if not AXIsProcessTrusted():
                logger.warning(
                    "⚠️  Accessibility access is NOT enabled for this process. "
                    "Grant it in System Settings → Privacy & Security → Accessibility. "
                    "Without it, mouse clicks will silently fail."
                )
        except (ImportError, AttributeError):
            # pyobjc-framework-Quartz not installed or API unavailable;
            # the moveTo check above is the best we can do.
            pass

    # ------------------------------------------------------------------
    # Capture methods
    # ------------------------------------------------------------------

    def capture_pil(self) -> Image.Image:
        """Capture the screen (cropped to game window if bound) and return a PIL Image."""
        img = self._capture_full_screen()

        pw = self._game_window_physical
        if pw is not None:
            # Crop using physical-pixel bounds so the crop rectangle
            # maps correctly onto the physical-pixel capture.
            crop_box = (pw.x, pw.y, pw.x + pw.width, pw.y + pw.height)
            if self.debug:
                logger.info(
                    "🔍 [DEBUG-SCREEN] Cropping capture to game window "
                    "(physical px): %dx%d at (%d, %d) | image size: %dx%d",
                    pw.width, pw.height, pw.x, pw.y,
                    img.width, img.height,
                )
                if self._game_window is not None:
                    w = self._game_window
                    logger.info(
                        "🔍 [DEBUG-SCREEN] Logical window: %dx%d at (%d, %d) — "
                        "scale=%.4f",
                        w.width, w.height, w.x, w.y, self._scale,
                    )
            img = img.crop(crop_box)

        return img

    def _capture_full_screen(self) -> Image.Image:
        """Capture the whole screen (internal, no cropping)."""
        if self._mss_available:
            try:
                # Grab monitor 1 (primary)
                raw = self._mss.grab(self._mss.monitors[1])
                return Image.frombytes("RGB", raw.size, raw.rgb)
            except Exception as exc:
                logger.warning(f"mss capture failed ({exc}), falling back to pyautogui.")

        return self._capture_via_pyautogui()

    @staticmethod
    def _capture_via_pyautogui() -> Image.Image:
        """Capture using pyautogui only (no mss fallback inside this method)."""
        try:
            import pyautogui

            return pyautogui.screenshot()
        except Exception as exc:
            raise ScreenCaptureError(f"All capture backends failed: {exc}") from exc

    def capture_png_bytes(self) -> bytes:
        """Capture the screen and return PNG-encoded bytes."""
        img = self.capture_pil()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def capture_base64(self) -> str:
        """Capture the screen and return a base64-encoded PNG string.

        This is the format expected by the OpenAI vision API.
        """
        import base64

        return base64.b64encode(self.capture_png_bytes()).decode("ascii")

    # ------------------------------------------------------------------
    # Coordinate scaling & click
    # ------------------------------------------------------------------

    def scale_coordinates(self, x: float, y: float) -> tuple[int, int]:
        """Scale LLM-returned coordinates to absolute logical screen coords.

        The LLM returns coordinates in the *physical-pixel* space of the
        screenshot it received (the cropped window image).  Two adjustments
        are applied:

        1. **HiDPI / Retina scale** (``_scale``) — maps physical-pixel
           coordinates to logical points that pyautogui operates in.
        2. **Window offset** — adds the game window's screen position
           (in logical points) to produce absolute screen coordinates.

        Because the crop in ``capture_pil()`` now uses physical-pixel
        bounds, the cropped image the LLM sees directly represents the
        game's output at native resolution.  No separate window→game
        resolution scale is needed — the LLM coordinates are already
        in the game's pixel space.
        """
        # Map physical-pixel coords → logical points
        scaled_x = int(x * self._scale)
        scaled_y = int(y * self._scale)

        # Offset by game window position (logical points, since pyautogui
        # operates in logical space)
        win = self._game_window
        if win is not None:
            abs_x = scaled_x + win.x
            abs_y = scaled_y + win.y
            if self.debug:
                logger.info(
                    "🔍 [DEBUG-SCREEN] scale_coordinates: raw=(%.1f, %.1f) × "
                    "scale=%.4f → logical=(%d, %d) + window-offset(logical)=(%d, %d) "
                    "→ absolute=(%d, %d)",
                    x, y, self._scale,
                    scaled_x, scaled_y, win.x, win.y, abs_x, abs_y,
                )
            return (abs_x, abs_y)

        if self.debug:
            logger.info(
                "🔍 [DEBUG-SCREEN] scale_coordinates: raw=(%.1f, %.1f) × "
                "scale=%.4f → logical=(%d, %d) [full-screen, no offset]",
                x, y, self._scale, scaled_x, scaled_y,
            )
        return (scaled_x, scaled_y)

    def click(self, x: int, y: int) -> None:
        """Click at the given *absolute* logical screen coordinates.

        This is the single entry-point for mouse clicks in the harness.
        It centralises coordinate handling and platform-specific quirks
        (e.g. macOS accessibility, multi-monitor offsets).

        When a game window is bound, ``scale_coordinates()`` should have
        already added the window offset, so the (x, y) passed here are
        absolute screen coordinates.

        Before clicking, the game window is re-focused if a
        ``WindowTracker`` is available.

        Args:
            x: Absolute logical X coordinate (after ``scale_coordinates``).
            y: Absolute logical Y coordinate (after ``scale_coordinates``).
        """
        import pyautogui

        # Re-focus the game window before clicking
        if self._window_tracker is not None and self._game_window_physical is not None:
            if self.debug:
                logger.info(
                    "🔍 [DEBUG-SCREEN] Re-focusing game window before click (%d, %d)",
                    x, y,
                )
            self._window_tracker.focus(self._game_window)
            # Small delay to let the window manager complete the focus
            import time
            time.sleep(0.15)

        # Re-check accessibility before each click when debugging
        if self.debug and sys.platform == "darwin":
            try:
                pos = pyautogui.position()
                pyautogui.moveTo(pos.x, pos.y)
                logger.info("🔍 [DEBUG-SCREEN] Accessibility re-check OK — can move mouse.")
            except Exception as exc:
                logger.warning(
                    "🔍 [DEBUG-SCREEN] Accessibility check FAILED before click: %s — "
                    "click may silently fail!", exc,
                )

        # Log current position for debugging click misalignment
        pos_before = pyautogui.position()

        if self.debug:
            logger.info(
                "🔍 [DEBUG-SCREEN] About to click(%d, %d) — mouse currently at (%d, %d)",
                x, y, pos_before.x, pos_before.y,
            )

        pyautogui.click(x, y)

        pos_after = pyautogui.position()
        logger.debug(
            "🖱️  click(%d, %d) — mouse was at (%d, %d), moved to (%d, %d)",
            x, y, pos_before.x, pos_before.y, pos_after.x, pos_after.y,
        )

        if self.debug:
            logger.info(
                "🔍 [DEBUG-SCREEN] click(%d, %d) completed — mouse now at (%d, %d)",
                x, y, pos_after.x, pos_after.y,
            )