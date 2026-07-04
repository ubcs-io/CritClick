"""Screen capture utilities for the vision harness.

Uses ``mss`` (fast, cross-platform) as the primary capture method, with
``pyautogui`` as a cross-platform fallback. On Linux, ``mss`` uses X11/shm
and works both on the local display and inside Xvfb virtual framebuffers.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from io import BytesIO

from PIL import Image

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
    """

    def __init__(self, scale: float = 0.0, debug: bool = False) -> None:
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

        # --- macOS accessibility check ----------------------------------------
        if sys.platform == "darwin":
            self._check_accessibility()

    @property
    def scale(self) -> float:
        """Effective coordinate scale factor (read-only)."""
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

        # Cross-platform heuristic: compare physical capture size to
        # pyautogui's reported screen size.
        try:
            import pyautogui

            pil_img = self._capture_via_pyautogui()
            logical_w, logical_h = pyautogui.size()
            phys_w, phys_h = pil_img.size
            if phys_w > 0 and phys_h > 0:
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
        ui_scale: float | None = None

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
        """Capture the whole screen and return a PIL Image."""
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
        """Scale LLM-returned coordinates by the effective scale factor.

        The LLM returns coordinates in the screenshot's physical pixel
        space.  On a Retina display the logical coordinate system is
        different, so we multiply by the scale factor to map to what
        ``pyautogui`` expects.
        """
        scaled_x = int(x * self._scale)
        scaled_y = int(y * self._scale)
        if self.debug:
            logger.info(
                "🔍 [DEBUG-SCREEN] scale_coordinates: raw=(%.1f, %.1f) × scale=%.4f → scaled=(%d, %d)",
                x, y, self._scale, scaled_x, scaled_y,
            )
        return (scaled_x, scaled_y)

    def click(self, x: int, y: int) -> None:
        """Click at the given *already-scaled* logical coordinates.

        This is the single entry-point for mouse clicks in the harness.
        It centralises coordinate handling and platform-specific quirks
        (e.g. macOS accessibility, multi-monitor offsets).

        Args:
            x: Logical X coordinate (after ``scale_coordinates``).
            y: Logical Y coordinate (after ``scale_coordinates``).
        """
        import pyautogui

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
