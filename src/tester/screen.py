"""Screen capture utilities for the vision harness.

Uses ``mss`` (fast, cross-platform) as the primary capture method, with
``pyautogui`` as a cross-platform fallback. On Linux, ``mss`` uses X11/shm
and works both on the local display and inside Xvfb virtual framebuffers.
"""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

logger = logging.getLogger("tester.screen")


class ScreenCaptureError(RuntimeError):
    """Raised when screen capture fails."""


class Capturer:
    """Captures the screen (or a region) and returns PNG bytes / base64.

    Supports coordinate scaling for DPI / resolution mismatches.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self._mss_available = False
        self._mss = None

        # Try to import mss (fast cross-platform capture)
        try:
            import mss

            self._mss = mss.MSS()
            self._mss_available = True
            logger.debug("Using mss for screen capture.")
        except ImportError:
            logger.debug("mss not available — falling back to pyautogui.")

    def capture_pil(self) -> Image.Image:
        """Capture the whole screen and return a PIL Image."""
        if self._mss_available:
            try:
                # Grab monitor 1 (primary)
                raw = self._mss.grab(self._mss.monitors[1])
                return Image.frombytes("RGB", raw.size, raw.rgb)
            except Exception as exc:
                logger.warning(f"mss capture failed ({exc}), falling back to pyautogui.")

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

    def scale_coordinates(self, x: float, y: float) -> tuple[int, int]:
        """Scale LLM-returned coordinates by the configured screen scale."""
        return (int(x * self.scale), int(y * self.scale))
