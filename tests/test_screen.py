"""Tests for screen capture, coordinate scaling, and click logic."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _suppress_accessibility_check():
    """Suppress the macOS accessibility check in *all* Capturer inits."""
    with patch.object(sys, "platform", "linux"):
        yield


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

class TestScaleDetection:
    def test_manual_scale_overrides_auto(self):
        from tester.screen import Capturer

        c = Capturer(scale=2.5)
        assert c.scale == 2.5

    def test_default_scale_is_one(self):
        from tester.screen import Capturer

        c = Capturer()
        assert c.scale == 1.0

    def test_retina_detection_parses_output(self):
        from tester.screen import Capturer

        fake_output = (
            "Displays:\n\n"
            "Color LCD:\n"
            "  Display Type: Built-in Retina LCD\n"
            "  Resolution: 3024 x 1964 Retina\n"
            "  UI Looks like: 1512 x 982 @ 2.00x\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = fake_output
            mock_run.return_value = mock_result
            detected = Capturer._detect_macos_retina_scale()

        assert detected == 2.0

    def test_retina_detection_no_match_returns_none(self):
        from tester.screen import Capturer

        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "No displays found"
            mock_run.return_value = mock_result
            detected = Capturer._detect_macos_retina_scale()

        assert detected is None

    def test_retina_detection_command_fails_returns_none(self):
        from tester.screen import Capturer

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            detected = Capturer._detect_macos_retina_scale()

        assert detected is None


# ---------------------------------------------------------------------------
# Coordinate scaling
# ---------------------------------------------------------------------------

class TestCoordinateScaling:
    def test_scale_coordinates_default(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        assert c.scale_coordinates(100, 200) == (100, 200)
        assert c.scale_coordinates(485.0, 425.0) == (485, 425)

    def test_scale_coordinates_retina(self):
        from tester.screen import Capturer

        c = Capturer(scale=0.5)
        assert c.scale_coordinates(100, 200) == (50, 100)
        assert c.scale_coordinates(485.0, 425.0) == (242, 212)

    def test_scale_coordinates_zero(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        assert c.scale_coordinates(0, 0) == (0, 0)


# ---------------------------------------------------------------------------
# Click
# ---------------------------------------------------------------------------

class TestClick:
    def test_click_calls_pyautogui_with_correct_coords(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        with patch("pyautogui.click") as mock_click, patch("pyautogui.position") as mock_pos:
            mock_pos.return_value = MagicMock(x=50, y=60)
            c.click(100, 200)

        mock_click.assert_called_once_with(100, 200)

    def test_click_logs_position_before(self, caplog):
        import logging

        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        with patch("pyautogui.click"), patch("pyautogui.position") as mock_pos:
            mock_pos.return_value = MagicMock(x=50, y=60)
            with caplog.at_level(logging.DEBUG, logger="tester.screen"):
                c.click(100, 200)

        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("click(100, 200)" in msg for msg in debug_messages)


# ---------------------------------------------------------------------------
# Accessibility check
# ---------------------------------------------------------------------------

class TestAccessibilityCheck:
    def test_accessibility_check_warns_when_move_fails(self, caplog, monkeypatch):
        from tester.screen import Capturer

        # Need to be on darwin for _check_accessibility to run
        monkeypatch.setattr(sys, "platform", "darwin")

        with patch("pyautogui.position", side_effect=Exception("permission denied")):
            Capturer._check_accessibility()

        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("Accessibility" in msg for msg in warning_messages)

    def test_accessibility_check_warns_with_ax_api(self, caplog, monkeypatch):
        from tester.screen import Capturer

        monkeypatch.setattr(sys, "platform", "darwin")

        # The static method does ``from ApplicationServices import AXIsProcessTrusted``
        # at runtime.  Pre-load a mock into sys.modules so the import succeeds.
        mock_ax = MagicMock()
        mock_ax.AXIsProcessTrusted = MagicMock(return_value=False)

        with patch("pyautogui.position", return_value=MagicMock(x=0, y=0)), \
             patch("pyautogui.moveTo"), \
             patch.dict("sys.modules", {"ApplicationServices": mock_ax}):
            Capturer._check_accessibility()

        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("Accessibility" in msg for msg in warning_messages)


# ---------------------------------------------------------------------------
# Capture backends
# ---------------------------------------------------------------------------

class TestCaptureBackends:
    def test_capture_pil_uses_mss_when_available(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c._mss_available = True

        fake_img = MagicMock()
        fake_raw = MagicMock()
        fake_raw.size = (1920, 1080)
        fake_raw.rgb = b"\x00" * 1920 * 1080 * 3

        with patch.object(c._mss, "grab", return_value=fake_raw), \
             patch("PIL.Image.frombytes", return_value=fake_img):
            result = c.capture_pil()
            assert result is fake_img

    def test_capture_pil_falls_back_to_pyautogui(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c._mss_available = True

        with patch.object(c._mss, "grab", side_effect=Exception("mss failed")), \
             patch.object(c, "_capture_via_pyautogui", return_value=MagicMock()) as mock_fallback:
            c.capture_pil()
            mock_fallback.assert_called_once()

    def test_capture_pil_uses_pyautogui_when_mss_unavailable(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c._mss_available = False

        with patch.object(c, "_capture_via_pyautogui", return_value=MagicMock()) as mock_pa:
            c.capture_pil()
            mock_pa.assert_called_once()


class TestCoordinateGrid:
    """Tests for the model-facing coordinate-grid overlay."""

    def test_grid_preserves_size_and_does_not_mutate(self):
        from PIL import Image

        from tester.screen import Capturer

        src = Image.new("RGB", (400, 300), (10, 20, 30))
        before = src.tobytes()

        out = Capturer.draw_coordinate_grid(src, spacing=100)

        assert out.size == src.size
        assert out is not src
        # Original image is untouched
        assert src.tobytes() == before

    def test_grid_draws_lines(self):
        from PIL import Image

        from tester.screen import Capturer

        src = Image.new("RGB", (400, 300), (10, 20, 30))
        out = Capturer.draw_coordinate_grid(src, spacing=100).convert("RGB")

        # Some pixels must differ from the flat background (lines + labels drawn)
        assert out.tobytes() != src.convert("RGB").tobytes()


class TestCalibrationTransform:
    """Tests for the model-coordinate → pixel-space transform."""

    def test_draw_calibration_markers(self):
        from tester.screen import Capturer

        img, markers = Capturer.draw_calibration_markers((1000, 500))

        assert img.size == (1000, 500)
        assert [m[0] for m in markers] == [1, 2, 3, 4]
        # Default 20%/80% quadrant positions
        assert (1, 200, 100) in markers
        assert (4, 800, 400) in markers

    def test_identity_when_unconfigured(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        # No calibration, no coordinate_max → coords are raw pixels
        assert c.scale_coordinates(123, 456) == (123, 456)

    def test_coordinate_max_normalizes(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c.set_coordinate_max(1000)
        c._last_capture_size = (1578, 916)

        x, y = c.scale_coordinates(500, 500)
        assert x == int(500 / 1000 * 1578)  # 789
        assert y == int(500 / 1000 * 916)   # 458

    def test_calibration_affine_applied(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c.set_coordinate_calibration((1.578, 0.0, 0.916, 0.0))

        assert c.scale_coordinates(100, 100) == (int(1.578 * 100), int(0.916 * 100))

    def test_calibration_takes_precedence_over_max(self):
        from tester.screen import Capturer

        c = Capturer(scale=1.0)
        c.set_coordinate_max(1000)
        c._last_capture_size = (1578, 916)
        c.set_coordinate_calibration((2.0, 0.0, 2.0, 0.0))

        assert c.scale_coordinates(10, 10) == (20, 20)