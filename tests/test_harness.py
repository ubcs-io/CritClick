"""Tests for the Harness action execution and bounds validation."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tester.client import LLMClient
from tester.harness import Harness
from tester.models import ActionResponse


class TestCoordinateValidation:
    def test_valid_coordinates(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        # Validate against the real screen size — coordinates inside screen bounds.
        # (100, 200) is trivially inside any real screen.
        assert harness._validate_coordinates(100, 200) is True
        assert harness._validate_coordinates(0, 0) is True

    def test_negative_within_margin(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        # margin is 50 — slightly negative coords are OK
        assert harness._validate_coordinates(-50, -50) is True

    def test_far_outside_bounds(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        # -100 is more than 50 margin below 0
        assert harness._validate_coordinates(-100, 100) is False
        assert harness._validate_coordinates(100, -100) is False
        # 50000 is far beyond any real screen
        assert harness._validate_coordinates(50000, 100) is False
        assert harness._validate_coordinates(100, 50000) is False

    def test_beyond_margin(self, sample_settings):
        """Validate against a mocked screen size for deterministic behaviour."""
        from unittest.mock import patch

        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        # Mock pyautogui.size() to return 1280×720 so the test is deterministic.
        with patch("tester.harness.pyautogui.size", return_value=(1280, 720)):
            # margin 50: valid range is -50 ≤ x ≤ 1330, -50 ≤ y ≤ 770
            assert harness._validate_coordinates(1280, 720) is True  # edge
            assert harness._validate_coordinates(1330, 770) is True  # exact margin edge
            assert harness._validate_coordinates(1331, 100) is False  # beyond margin
            assert harness._validate_coordinates(100, 771) is False   # beyond margin


class TestActionExecution:
    @pytest.fixture
    def dry_run_harness(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        # Mock the capturer to avoid real screen interactions
        harness.capturer = MagicMock()
        harness.capturer.scale = 1.0
        harness.capturer.scale_coordinates.side_effect = lambda x, y: (int(x), int(y))
        return harness

    def test_done_action(self, dry_run_harness):
        resp = ActionResponse(
            description="Credits",
            action="done",
            reasoning="Game ended",
            narrative="Game over",
        )
        result = dry_run_harness._execute(resp)
        assert result == "completed"

    def test_wait_action(self, dry_run_harness):
        resp = ActionResponse(
            description="Loading",
            action="wait",
            reasoning="Scene transition",
            narrative="Waiting",
        )
        result = dry_run_harness._execute(resp)
        assert result == "waited"

    def test_click_action_valid(self, dry_run_harness):
        resp = ActionResponse(
            description="Button",
            action="click",
            coordinates=[100.0, 200.0],
            reasoning="Need to click",
            narrative="Clicked",
        )
        result = dry_run_harness._execute(resp)
        assert result == "clicked"
        dry_run_harness.capturer.scale_coordinates.assert_called_once_with(100.0, 200.0)

    def test_click_action_valid_not_dry_run_calls_click(self, sample_settings):
        """When not in dry-run, capturer.click() should be called with scaled coords."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=False)
        harness.capturer = MagicMock()
        harness.capturer.scale = 0.5
        harness.capturer.scale_coordinates.return_value = (50, 100)

        resp = ActionResponse(
            description="Button",
            action="click",
            coordinates=[100.0, 200.0],
            reasoning="Need to click",
            narrative="Clicked",
        )
        result = harness._execute(resp)
        assert result == "clicked"
        harness.capturer.scale_coordinates.assert_called_once_with(100.0, 200.0)
        harness.capturer.click.assert_called_once_with(50, 100)

    def test_click_no_coordinates(self, dry_run_harness):
        resp = ActionResponse(
            description="Button",
            action="click",
            coordinates=[],
            reasoning="Need to click",
            narrative="Clicked",
        )
        result = dry_run_harness._execute(resp)
        assert result == "unknown"

    def test_click_out_of_bounds(self, dry_run_harness):
        resp = ActionResponse(
            description="Button",
            action="click",
            coordinates=[5000.0, 5000.0],
            reasoning="Need to click",
            narrative="Clicked",
        )
        result = dry_run_harness._execute(resp)
        assert result == "unknown"

    def test_type_action(self, dry_run_harness):
        resp = ActionResponse(
            description="Input",
            action="type",
            text_to_type="hello world",
            reasoning="Need to type",
            narrative="Typed",
        )
        result = dry_run_harness._execute(resp)
        assert result == "typed"

    def test_press_action(self, dry_run_harness):
        resp = ActionResponse(
            description="Dialogue",
            action="press",
            key_to_press="enter",
            reasoning="Advance dialogue",
            narrative="Pressed enter",
        )
        result = dry_run_harness._execute(resp)
        assert result == "pressed"

    def test_press_no_key(self, dry_run_harness):
        resp = ActionResponse(
            description="Dialogue",
            action="press",
            reasoning="Advance dialogue",
            narrative="Pressed",
        )
        result = dry_run_harness._execute(resp)
        assert result == "unknown"


class TestStuckDetection:
    def test_stuck_counter_increments(self, sample_settings):
        """Counter tracks repeats: first occurrence=0, first repeat=1, etc."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)

        resp = ActionResponse(
            description="x",
            action="wait",
            reasoning="x",
            narrative="x",
        )
        harness._execute(resp)
        assert harness.stuck_counter == 0  # first occurrence
        harness._execute(resp)
        assert harness.stuck_counter == 1  # first repeat
        harness._execute(resp)
        assert harness.stuck_counter == 2  # second repeat

    def test_stuck_counter_resets_on_new_action(self, sample_settings):
        """Switching to a different action resets the counter to 0."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)

        wait_resp = ActionResponse(
            description="x", action="wait", reasoning="x", narrative="x"
        )
        click_resp = ActionResponse(
            description="x", action="click", coordinates=[1, 1], reasoning="x", narrative="x"
        )
        harness._execute(wait_resp)
        harness._execute(wait_resp)
        assert harness.stuck_counter == 1  # one repeat
        harness._execute(click_resp)
        assert harness.stuck_counter == 0  # reset by new action


class TestRunSummary:
    def test_action_counts_tracked(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)

        responses = [
            ActionResponse(description="x", action="click", coordinates=[1, 1], reasoning="x", narrative="x"),
            ActionResponse(description="x", action="click", coordinates=[1, 1], reasoning="x", narrative="x"),
            ActionResponse(description="x", action="wait", reasoning="x", narrative="x"),
            ActionResponse(description="x", action="done", reasoning="x", narrative="x"),
        ]
        for resp in responses:
            harness._execute(resp)

        assert harness.action_counts.get("click") == 2
        assert harness.action_counts.get("wait") == 1
        assert harness.action_counts.get("done") == 1


# ---------------------------------------------------------------------------
# Recap tests
# ---------------------------------------------------------------------------


class TestRunRecap:
    """Tests for the _run_recap() post-run LLM feedback feature."""

    @pytest.fixture
    def recap_harness(self, sample_settings):
        """Harness with a mocked LLM client and a temp JSONL log file."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_settings.logging.save_screenshots = False

            harness = Harness(sample_settings, launcher, client, dry_run=True, run_id="test-run-12345")
            harness.start_time = harness.start_time  # set by __init__

            # _apply_run_namespace() rewrites log_file to runs/<run_id>/...
            # so we must write the log there, not to our original temp path.
            log_file = harness.settings.logging.log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Write a minimal JSONL log so _run_recap() has data to read
            entries = [
                {
                    "step": 1,
                    "timestamp": "2025-01-01T00:00:00",
                    "llm_response": {
                        "description": "Main menu",
                        "action": "click",
                        "coordinates": [100, 200],
                        "reasoning": "Start button visible",
                        "narrative": "Clicked start button.",
                    },
                    "duration_ms": 500,
                },
                {
                    "step": 2,
                    "timestamp": "2025-01-01T00:00:01",
                    "llm_response": {
                        "description": "Dialogue",
                        "action": "press",
                        "key_to_press": "enter",
                        "reasoning": "Advance dialogue",
                        "narrative": "Advanced dialogue.",
                    },
                    "duration_ms": 400,
                },
                {
                    "step": 3,
                    "timestamp": "2025-01-01T00:00:02",
                    "llm_response": {
                        "description": "Credits",
                        "action": "done",
                        "reasoning": "Game finished",
                        "narrative": "Game ended.",
                    },
                    "duration_ms": 300,
                },
            ]
            with open(log_file, "w", encoding="utf-8") as fh:
                for entry in entries:
                    fh.write(json.dumps(entry) + "\n")

            # Simulate a completed run
            harness.step = 3
            harness.completion_reason = "completed"
            harness.action_counts = {"click": 1, "press": 1, "done": 1}

            yield harness, tmpdir

    def test_recap_returns_dict_on_success(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.analyze.return_value = {
            "key_complaint": "The game had no save button in the options menu."
        }

        result = harness._run_recap()
        assert result is not None
        assert result["key_complaint"] == "The game had no save button in the options menu."

    def test_recap_calls_llm_with_recap_prompt(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.analyze.return_value = {
            "key_complaint": "Everything worked fine."
        }

        harness._run_recap()
        mock_client.analyze.assert_called_once()
        _, system_prompt, user_prompt = mock_client.analyze.call_args[0]
        assert "complaint" in system_prompt.lower()
        assert "roadblock" in system_prompt.lower()
        assert "Step 1" in user_prompt
        assert "Step 2" in user_prompt
        assert "Step 3" in user_prompt
        assert "completed" in user_prompt
        assert "click=1" in user_prompt

    def test_recap_returns_none_on_llm_error(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.analyze.side_effect = RuntimeError("API down")

        result = harness._run_recap()
        assert result is None

    def test_recap_returns_none_on_validation_error(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.analyze.return_value = {"wrong_field": "bad"}

        result = harness._run_recap()
        assert result is None

    def test_recap_skipped_when_no_steps(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "playthrough_log.jsonl")
            sample_settings.logging.log_file = log_file

            harness = Harness(sample_settings, launcher, client, dry_run=True)
            harness.step = 0
            result = harness._run_recap()
            assert result is None

    def test_recap_skipped_when_no_log_file(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        harness = Harness(sample_settings, launcher, client, dry_run=True)
        harness.settings.logging.log_file = "/nonexistent/path/log.jsonl"
        harness.step = 5
        result = harness._run_recap()
        assert result is None

    def test_recap_skipped_when_empty_log_file(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "playthrough_log.jsonl")
            sample_settings.logging.log_file = log_file
            # Create empty file
            Path(log_file).touch()

            harness = Harness(sample_settings, launcher, client, dry_run=True)
            harness.step = 3
            result = harness._run_recap()
            assert result is None


class TestCoordinateGridApplication:
    """The coordinate grid should be applied to the model image per config."""

    def _make_harness(self, sample_settings, grid_enabled):
        from PIL import Image

        sample_settings.harness.coordinate_grid = grid_enabled
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        # Return a valid 'wait' action so analyze_and_act has no click side effects
        client.analyze.return_value = {
            "description": "menu",
            "action": "wait",
            "reasoning": "let scene settle",
            "narrative": "waited",
        }
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        harness.capturer = MagicMock()
        harness.capturer.scale = 1.0
        harness.capturer.game_window = None
        harness.capturer.capture_pil.return_value = Image.new("RGB", (200, 150), (0, 0, 0))
        return harness

    def test_grid_applied_when_enabled(self, sample_settings):
        from unittest.mock import patch

        harness = self._make_harness(sample_settings, grid_enabled=True)
        with patch("tester.harness.Capturer.draw_coordinate_grid") as mock_grid:
            from PIL import Image
            mock_grid.return_value = Image.new("RGB", (200, 150), (1, 1, 1))
            harness.analyze_and_act()
        mock_grid.assert_called_once()
        assert mock_grid.call_args.kwargs["spacing"] == sample_settings.harness.grid_spacing

    def test_grid_skipped_when_disabled(self, sample_settings):
        from unittest.mock import patch

        harness = self._make_harness(sample_settings, grid_enabled=False)
        with patch("tester.harness.Capturer.draw_coordinate_grid") as mock_grid:
            harness.analyze_and_act()
        mock_grid.assert_not_called()
