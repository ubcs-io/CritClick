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

from PIL import Image, ImageDraw, ImageFont

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

        # --- Model coordinate-space handling ----------------------------------
        # Size (physical px) of the most recent capture — the pixel space the
        # model's coordinates should map into.
        self._last_capture_size: tuple[int, int] | None = None
        # Config fallback: if > 0, the model reports coordinates normalized to
        # [0, coordinate_max] per axis (e.g. 1000 for Qwen-VL).
        self._coordinate_max: float = 0.0
        # Per-axis affine transform (a_x, b_x, a_y, b_y) from model space to
        # capture-pixel space, learned by the calibration probe. Takes
        # precedence over _coordinate_max when set.
        self._calibration: tuple[float, float, float, float] | None = None

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
    def last_capture_size(self) -> tuple[int, int] | None:
        """Physical-pixel size of the most recent capture, or ``None``."""
        return self._last_capture_size

    def set_coordinate_max(self, coordinate_max: float) -> None:
        """Set the config fallback for normalized model coordinates.

        ``coordinate_max`` is the per-axis range the model normalizes to
        (e.g. ``1000`` for Qwen-VL). ``0`` means coordinates are raw pixels.
        """
        self._coordinate_max = max(0.0, float(coordinate_max))

    def set_coordinate_calibration(
        self, calibration: tuple[float, float, float, float] | None
    ) -> None:
        """Set the learned affine transform ``(a_x, b_x, a_y, b_y)``.

        Maps model-space coordinates into capture-pixel space via
        ``pixel = a * model + b`` per axis. ``None`` clears it (falls back to
        ``coordinate_max`` or identity).
        """
        self._calibration = calibration
        if self.debug and calibration is not None:
            logger.info(
                "🔍 [DEBUG-SCREEN] Coordinate calibration set: "
                "x_px=%.4f·x+%.1f | y_px=%.4f·y+%.1f",
                calibration[0], calibration[1], calibration[2], calibration[3],
            )

    def _to_pixel_space(self, x: float, y: float) -> tuple[float, float]:
        """Map model-reported coords into capture-pixel space.

        Applies, in precedence order: the learned calibration transform, then
        the ``coordinate_max`` normalization fallback, else identity (the model
        already reports pixels).
        """
        if self._calibration is not None:
            ax, bx, ay, by = self._calibration
            return (ax * x + bx, ay * y + by)
        if self._coordinate_max > 0 and self._last_capture_size is not None:
            w, h = self._last_capture_size
            return (x / self._coordinate_max * w, y / self._coordinate_max * h)
        return (x, y)

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

            # Diagnostic: compare mss monitor info vs pyautogui vs full-screen capture.
            # This is critical for understanding coordinate misalignment root causes.
            mss_monitor_info = "N/A"
            if self._mss_available and self._mss is not None:
                try:
                    m1 = self._mss.monitors[1]
                    mss_monitor_info = (
                        f"mss.monitors[1]={{{m1['width']}x{m1['height']}+{m1['left']}+{m1['top']}}}"
                    )
                except Exception:
                    pass

            if phys_w > 0 and phys_h > 0 and logical_w > 0 and logical_h > 0:
                scale_w = logical_w / phys_w
                scale_h = logical_h / phys_h
                avg = (scale_w + scale_h) / 2.0
                logger.info(
                    "🧮 Resolution comparison: pyautogui.size()=%dx%d | "
                    "capture(pixels)=%dx%d | %s → scale_w=%.4f scale_h=%.4f avg=%.4f",
                    logical_w, logical_h, phys_w, phys_h, mss_monitor_info,
                    scale_w, scale_h, avg,
                )
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

        # Remember the pixel space the model will report coordinates in.
        self._last_capture_size = img.size
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

    def _diagnose_window_position(self, click_x: int, click_y: int) -> None:
        """Re-query the game window position and compare against cached bounds.

        Logs a warning if the window has moved since it was first detected,
        which would explain click misalignment.  Called before each click
        when debug is enabled on macOS.
        """
        if self._window_tracker is None:
            return

        import Quartz  # type: ignore[import-untyped]

        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID,
        )
        pid = self._window_tracker._pid
        for entry in window_list:
            if entry.get(Quartz.kCGWindowOwnerPID) != pid:
                continue
            if entry.get(Quartz.kCGWindowLayer, 99) != 0:
                continue
            bounds = entry.get(Quartz.kCGWindowBounds, {})
            cur_x = int(bounds.get("X", 0))
            cur_y = int(bounds.get("Y", 0))
            cur_w = int(bounds.get("Width", 0))
            cur_h = int(bounds.get("Height", 0))
            if cur_w <= 0 or cur_h <= 0:
                continue

            cached = self._game_window
            if cached is not None:
                dx = cur_x - cached.x
                dy = cur_y - cached.y
                dw = cur_w - cached.width
                dh = cur_h - cached.height
                if dx != 0 or dy != 0 or dw != 0 or dh != 0:
                    logger.warning(
                        "🔍 [DEBUG-SCREEN] Window POSITION DRIFT detected! "
                        "cached=+%d+%d %dx%d | current=+%d+%d %dx%d | "
                        "delta=(%+d, %+d, %+d, %+d) | "
                        "click-target=absolute(%d, %d)",
                        cached.x, cached.y, cached.width, cached.height,
                        cur_x, cur_y, cur_w, cur_h,
                        dx, dy, dw, dh,
                        click_x, click_y,
                    )
                    # Recompute where the click would land in the *current*
                    # game-window-relative space for comparison
                    if cached.x != 0:
                        rel_x = click_x - cur_x
                        rel_y = click_y - cur_y
                        logger.warning(
                            "🔍 [DEBUG-SCREEN] If window were still at cached position, "
                            "click would be at window-relative=(%d,%d) but with current "
                            "position it lands at window-relative=(%d,%d) [window=%dx%d]",
                            click_x - cached.x, click_y - cached.y,
                            rel_x, rel_y, cur_w, cur_h,
                        )
                else:
                    logger.info(
                        "🔍 [DEBUG-SCREEN] Window position stable: +%d+%d %dx%d — "
                        "click-target=absolute(%d, %d) → window-relative=(%d, %d)",
                        cur_x, cur_y, cur_w, cur_h,
                        click_x, click_y,
                        click_x - cur_x, click_y - cur_y,
                    )
            break

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

        A preliminary step maps the model's coordinate convention (raw
        pixels, normalized [0, coordinate_max], or a calibrated affine
        transform) into that capture-pixel space via ``_to_pixel_space()``.
        """
        # Map model-space coords → capture-pixel space (calibration / normalization)
        px, py = self._to_pixel_space(x, y)
        if self.debug and (px != x or py != y):
            logger.info(
                "🔍 [DEBUG-SCREEN] model-space (%.1f, %.1f) → capture-pixel (%.1f, %.1f)",
                x, y, px, py,
            )

        # Map physical-pixel coords → logical points
        scaled_x = int(px * self._scale)
        scaled_y = int(py * self._scale)

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

        # Re-check the window position in case it was moved since startup.
        # This diagnostic helps confirm the cached offset is still correct.
        if self.debug and self._window_tracker is not None and sys.platform == "darwin":
            self._diagnose_window_position(x, y)

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

    # ------------------------------------------------------------------
    # Debug overlay drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_overlay_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
        """Return ``(font, font_small)`` for overlay drawing.

        Tries a few common system fonts and falls back to PIL's built-in
        bitmap font.  Shared by ``draw_debug_overlay`` and
        ``draw_coordinate_grid`` so the font selection lives in one place.
        """
        for font_name in (
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "arial.ttf",
            "Arial.ttf",
        ):
            try:
                return (
                    ImageFont.truetype(font_name, 18),
                    ImageFont.truetype(font_name, 14),
                )
            except Exception:
                continue
        default = ImageFont.load_default()
        return (default, default)

    @staticmethod
    def draw_coordinate_grid(
        image: Image.Image,
        *,
        spacing: int = 100,
    ) -> Image.Image:
        """Overlay a labeled coordinate grid on a copy of *image*.

        Thin, semi-transparent gridlines are drawn every ``spacing`` pixels.
        Each line is labeled with its pixel value: x-values along the top and
        bottom edges, y-values along the left and right edges, so every line
        is anchored on both sides for mid-screen interpolation.

        The grid gives non-grounding vision models (e.g. gpt-4o) explicit
        visual reference lines to read click coordinates off of.  Because the
        labels are drawn at true pixel positions and scale with the image, the
        reference survives any internal downscaling the model performs.

        The original *image* is **not** modified; a new RGBA image is returned.

        Args:
            image: The captured game-window screenshot (PIL Image).
            spacing: Pixel distance between adjacent gridlines.

        Returns:
            A new ``PIL.Image.Image`` with the grid drawn on top.
        """
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        _, font_small = Capturer._load_overlay_fonts()

        w, h = base.size
        spacing = max(int(spacing), 1)
        line_color = (0, 200, 255, 90)
        label_color = (0, 220, 255, 255)
        label_bg = (0, 0, 0, 170)

        def _label(x: int, y: int, text: str, anchor: str) -> None:
            """Draw *text* with a dark backing rect. *anchor* controls alignment.

            Horizontal: ``left`` / ``center`` / ``right``.
            Vertical:   ``top``  / ``middle`` / ``bottom``.
            Labels are anchored so the number straddles its own gridline
            (centered across the line) rather than sitting offset to one side —
            offset labels bias the reader toward the low-x / high-y corner.
            """
            bbox = draw.textbbox((0, 0), text, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if "right" in anchor:
                x -= tw
            elif "center" in anchor:
                x -= tw // 2
            if "bottom" in anchor:
                y -= th
            elif "middle" in anchor:
                y -= th // 2
            # Clamp so labels stay fully inside the image
            x = max(0, min(x, w - tw))
            y = max(0, min(y, h - th))
            draw.rectangle([x - 2, y - 1, x + tw + 2, y + th + 1], fill=label_bg)
            draw.text((x, y), text, fill=label_color, font=font_small)

        # Vertical lines + x labels centered on the line (top and bottom edges)
        for gx in range(0, w, spacing):
            draw.line([(gx, 0), (gx, h)], fill=line_color, width=1)
            _label(gx, 1, str(gx), "center-top")
            _label(gx, h - 1, str(gx), "center-bottom")

        # Horizontal lines + y labels centered on the line (left and right edges)
        for gy in range(0, h, spacing):
            draw.line([(0, gy), (w, gy)], fill=line_color, width=1)
            _label(1, gy, str(gy), "left-middle")
            _label(w - 1, gy, str(gy), "right-middle")

        return Image.alpha_composite(base, overlay)

    @staticmethod
    def draw_calibration_markers(
        size: tuple[int, int],
        *,
        fractions: tuple[tuple[float, float], ...] = (
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        ),
    ) -> tuple[Image.Image, list[tuple[int, int, int]]]:
        """Render a calibration probe image with numbered crosshair markers.

        Markers are placed at *fractions* of the image size (default: one per
        quadrant at 20%/80%). Each is a bold crosshair with a number label.

        Args:
            size: ``(width, height)`` of the probe image, which MUST match the
                real capture size so the model's coordinate convention (which
                can depend on image dimensions) is identical.
            fractions: ``(fx, fy)`` positions of each marker, in [0, 1].

        Returns:
            ``(image, markers)`` where *markers* is a list of
            ``(label, x, y)`` known center positions in image-pixel space.
        """
        w, h = size
        img = Image.new("RGBA", (w, h), (24, 26, 32, 255))
        draw = ImageDraw.Draw(img)
        font, _ = Capturer._load_overlay_fonts()

        markers: list[tuple[int, int, int]] = []
        arm = max(12, min(w, h) // 40)
        color = (255, 60, 60, 255)
        for i, (fx, fy) in enumerate(fractions, start=1):
            cx = int(round(fx * w))
            cy = int(round(fy * h))
            draw.line([(cx - arm, cy), (cx + arm, cy)], fill=color, width=3)
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=color, width=3)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
            # Label offset toward image center so it never clips the edge
            lx = cx + (arm + 6 if fx < 0.5 else -arm - 22)
            ly = cy + (arm + 4 if fy < 0.5 else -arm - 22)
            draw.text((lx, ly), str(i), fill=(255, 255, 255, 255), font=font)
            markers.append((i, cx, cy))

        return img.convert("RGB"), markers

    @staticmethod
    def draw_debug_overlay(
        image: Image.Image,
        *,
        bounding_box: list[float] | None = None,
        click_point: tuple[int, int] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Image.Image:
        """Draw debug annotations on a copy of *image* and return it.

        Annotations drawn (when the relevant data is provided):

        * **bounding_box** — green rectangle with label
        * **click_point** — lime crosshair (10 px arms)
        * **metadata** — semi-transparent panel in the top‑left corner
          with each key: value on its own line

        The original *image* is **not** modified; a new RGBA image is
        returned.

        Args:
            image: The captured game‑window screenshot (PIL Image).
            bounding_box: ``[x1, y1, x2, y2]`` in image‑pixel coords.
            click_point: ``(cx, cy)`` centre point that was/would be clicked.
            metadata: Dict of string labels → string values to display.

        Returns:
            A new ``PIL.Image.Image`` with annotations drawn on top.
        """
        # Work on an RGBA copy so we can use transparency
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # --- font selection ---
        font, font_small = Capturer._load_overlay_fonts()

        # --- bounding box ---
        if bounding_box is not None and len(bounding_box) == 4:
            x1, y1, x2, y2 = [int(v) for v in bounding_box]
            # Normalise order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 220), width=3)
            label = f"bbox [{x1},{y1},{x2},{y2}]"
            # Place label above the box if there's room, otherwise inside
            label_y = y1 - 22 if y1 >= 24 else y1 + 4
            # Draw a background rectangle behind the label for readability
            bbox_text = draw.textbbox((x1, label_y), label, font=font_small)
            draw.rectangle(
                [bbox_text[0] - 3, bbox_text[1] - 1, bbox_text[2] + 3, bbox_text[3] + 1],
                fill=(0, 0, 0, 180),
            )
            draw.text((x1, label_y), label, fill=(0, 255, 0, 255), font=font_small)

        # --- click point crosshair ---
        if click_point is not None:
            cx, cy = click_point
            lime = (50, 255, 50, 240)
            arm = 12
            draw.line([(cx - arm, cy), (cx + arm, cy)], fill=lime, width=2)
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=lime, width=2)
            # Small centre dot
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=lime)

        # --- metadata panel ---
        if metadata:
            lines = [f"{k}: {v}" for k, v in metadata.items()]
            if lines:
                # Measure the widest line to size the panel
                max_width = 0
                line_heights: list[tuple[int, int, int, int]] = []
                for line in lines:
                    bbox_t = draw.textbbox((0, 0), line, font=font_small)
                    line_heights.append(bbox_t)
                    w = bbox_t[2] - bbox_t[0]
                    if w > max_width:
                        max_width = w
                line_h = line_heights[0][3] - line_heights[0][1] + 4
                panel_h = len(lines) * line_h + 12
                panel_w = max_width + 20

                # Draw semi-transparent panel background
                draw.rectangle(
                    [6, 6, 6 + panel_w, 6 + panel_h],
                    fill=(0, 0, 0, 175),
                    outline=(255, 255, 255, 100),
                    width=1,
                )

                for i, line in enumerate(lines):
                    draw.text(
                        (16, 12 + i * line_h),
                        line,
                        fill=(255, 255, 255, 240),
                        font=font_small,
                    )

        # Composite overlay onto base
        result = Image.alpha_composite(base, overlay)
        return result