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
    """Tests for the _llm_recap() post-run LLM feedback feature."""

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
        mock_client.summarize.return_value = {
            "next_action": "Add a save button to the options menu.",
            "summary": "The game had no save button in the options menu.",
            "related_steps": [2],
            "persona_reviews": [
                {
                    "persona": "Impatient completionist",
                    "experience": "By step 2 I'm racing ahead with no way to save.",
                    "friction": ["No save button at step 2"],
                    "sentiment": "frustrated",
                }
            ],
            "overall_verdict": "Flows well but lacks a save affordance.",
        }

        result = harness._llm_recap()
        assert result is not None
        assert result["next_action"] == "Add a save button to the options menu."
        assert result["related_steps"] == [2]
        assert result["persona_reviews"][0]["persona"] == "Impatient completionist"
        assert result["overall_verdict"]

    def test_recap_calls_llm_with_recap_prompt(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.summarize.return_value = {
            "next_action": "No action needed — everything worked fine.",
        }

        harness._llm_recap()
        mock_client.summarize.assert_called_once()
        _, system_prompt, user_prompt = mock_client.summarize.call_args[0]
        assert "next_action" in system_prompt
        assert "Step 1" in user_prompt
        assert "Step 2" in user_prompt
        assert "Step 3" in user_prompt
        assert "completed" in user_prompt
        assert "click=1" in user_prompt
        # No personas configured → the prompt asks for a single "Typical player".
        assert "Typical player" in user_prompt
        # The recap is given its own larger token budget.
        assert mock_client.summarize.call_args.kwargs["max_tokens"] >= 1500

    def test_recap_passes_configured_personas(self, recap_harness):
        harness, _tmpdir = recap_harness
        harness.settings.review.personas = ["Speedrunner", "Lore reader"]
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.summarize.return_value = {"next_action": "Ship it."}

        harness._llm_recap()
        _, _system_prompt, user_prompt = mock_client.summarize.call_args[0]
        assert "Speedrunner" in user_prompt
        assert "Lore reader" in user_prompt

    def test_recap_returns_none_on_llm_error(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.summarize.side_effect = RuntimeError("API down")

        result = harness._llm_recap()
        assert result is None

    def test_recap_returns_none_on_validation_error(self, recap_harness):
        harness, _tmpdir = recap_harness
        mock_client: MagicMock = harness.llm  # type: ignore[assignment]
        mock_client.summarize.return_value = {"wrong_field": "bad"}

        result = harness._llm_recap()
        assert result is None

    def test_recap_skipped_when_no_steps(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "playthrough_log.jsonl")
            sample_settings.logging.log_file = log_file

            harness = Harness(sample_settings, launcher, client, dry_run=True)
            harness.step = 0
            result = harness._llm_recap()
            assert result is None

    def test_recap_skipped_when_no_log_file(self, sample_settings):
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)

        harness = Harness(sample_settings, launcher, client, dry_run=True)
        harness.settings.logging.log_file = "/nonexistent/path/log.jsonl"
        harness.step = 5
        result = harness._llm_recap()
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
            result = harness._llm_recap()
            assert result is None


class TestNextActionRendering:
    """The persona review must render into NEXT_ACTION.md."""

    def test_write_next_action_renders_persona_sections(self, sample_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = Harness(
                sample_settings,
                MagicMock(),
                MagicMock(spec=LLMClient),
                dry_run=True,
                run_id="render-test",
                runs_dir=tmpdir,
            )
            report = {
                "next_action": "Fix the overlapping title text.",
                "summary": "Menu polish issues.",
                "persona_reviews": [
                    {
                        "persona": "Impatient completionist",
                        "experience": "I blaze through but hit a wall at step 3.",
                        "friction": ["Overlapping title text at step 1"],
                        "sentiment": "frustrated",
                    },
                    {
                        "persona": "First-time casual",
                        "experience": "I'm unsure where to click on the menu.",
                        "friction": [],
                        "sentiment": "confused",
                    },
                ],
                "overall_verdict": "Playable but rough around the menu.",
                "related_steps": [1, 3],
                "status": "completed",
                "severity": "info",
            }
            harness._write_next_action(report)

            content = (Path(tmpdir) / "render-test" / "NEXT_ACTION.md").read_text()
            assert "## Player Experience Review" in content
            assert "### Impatient completionist — _frustrated_" in content
            assert "- Overlapping title text at step 1" in content
            assert "### First-time casual — _confused_" in content
            assert "## Overall" in content
            assert "Playable but rough around the menu." in content


class TestBuildNextAction:
    """Tests for the hybrid _build_next_action() takeaway generator."""

    def _harness(self, sample_settings):
        return Harness(sample_settings, MagicMock(), MagicMock(spec=LLMClient), dry_run=True)

    def test_game_death_is_deterministic_and_names_file(self, sample_settings):
        harness = self._harness(sample_settings)
        harness.step = 5
        harness.completion_reason = "game_died"
        harness._exit_code = 3
        # Simulate a Ren'Py traceback captured from the game's stderr.
        stderr = (
            'Traceback (most recent call last):\n'
            '  File "game/script.rpy", line 42, in script\n'
            'Exception: image not found: bg office\n'
        )
        harness._error_source_file, harness._error_excerpt = harness._extract_error_source(stderr)

        report = harness._build_next_action()

        # No LLM call on the hard-error path.
        harness.llm.analyze.assert_not_called()
        assert report["status"] == "game_died"
        assert report["error_source_file"] == "game/script.rpy"
        assert report["related_steps"] == [5]
        assert "game/script.rpy" in report["next_action"]
        assert "image not found" in report["error_text"]

    def test_crash_captures_error_text(self, sample_settings):
        harness = self._harness(sample_settings)
        harness.step = 8
        harness.completion_reason = "crashed: boom"
        harness._exit_code = 2
        harness._crash_traceback = (
            'Traceback (most recent call last):\n'
            '  File "src/tester/harness.py", line 500, in analyze_and_act\n'
            'RuntimeError: boom\n'
        )
        harness._error_source_file, harness._error_excerpt = harness._extract_error_source(
            harness._crash_traceback
        )

        report = harness._build_next_action()
        assert report["status"] == "crashed"
        assert report["related_steps"] == [8]
        assert "RuntimeError: boom" in report["error_text"]

    def test_soft_completion_uses_fallback_when_llm_unavailable(self, sample_settings):
        harness = self._harness(sample_settings)
        harness.step = 3
        harness.completion_reason = "completed"
        harness.settings.logging.log_file = "/nonexistent/path/log.jsonl"

        report = harness._build_next_action()
        assert report["status"] == "completed"
        assert report["next_action"]  # non-empty deterministic fallback


class TestErrorContextCollection:
    """Game-side error context is drained from the launcher and prioritised."""

    def _harness(self, sample_settings, *, stdout=None, stderr=None, engine_log=None,
                 dry_run=False):
        launcher = MagicMock()
        launcher.get_stdout_tail.return_value = stdout
        launcher.get_stderr_tail.return_value = stderr
        launcher.read_engine_error_log.return_value = engine_log
        client = MagicMock(spec=LLMClient)
        return Harness(sample_settings, launcher, client, dry_run=dry_run)

    def test_engine_log_preferred_over_stderr(self, sample_settings):
        harness = self._harness(
            sample_settings,
            stderr='File "game/other.rpy", line 9\nException: noise\n',
            engine_log=(
                'Traceback (most recent call last):\n'
                '  File "game/script.rpy", line 42, in script\n'
                'Exception: image not found: bg office\n'
            ),
        )
        harness._collect_error_context()
        assert harness._error_source_file == "game/script.rpy"
        assert "image not found" in harness._error_excerpt

    def test_stderr_used_when_no_engine_log(self, sample_settings):
        harness = self._harness(
            sample_settings,
            stderr=(
                'Traceback (most recent call last):\n'
                '  File "game/menu.rpy", line 7, in script\n'
                'Exception: kaboom\n'
            ),
        )
        harness._collect_error_context()
        assert harness._error_source_file == "game/menu.rpy"
        assert "kaboom" in harness._error_excerpt

    def test_soft_run_ignores_benign_stderr(self, sample_settings):
        # A clean run whose engine printed only warnings must not be reported
        # as an error when an error signal is required.
        harness = self._harness(
            sample_settings,
            stderr="INFO: shader compiled\nWARNING: deprecated setting\n",
        )
        harness._collect_error_context(require_error_signal=True)
        assert harness._error_source_file is None
        assert harness._error_excerpt is None

    def test_hard_run_surfaces_stderr_without_classic_signal(self, sample_settings):
        # On a hard failure we surface the tail even if it lacks a traceback.
        harness = self._harness(sample_settings, stderr="Killed: 9\n")
        harness._collect_error_context(require_error_signal=False)
        assert harness._error_excerpt is not None
        assert "Killed" in harness._error_excerpt

    def test_godot_script_error_recognised_as_signal(self, sample_settings):
        harness = self._harness(
            sample_settings,
            stderr="SCRIPT ERROR: Invalid call. at: res://player.gd:88\n",
        )
        harness._collect_error_context(require_error_signal=True)
        assert harness._error_source_file == "res://player.gd:88"


class TestFinalErrorContext:
    """_gather_final_error_context wires collection + harness-crash fallback."""

    def test_crash_falls_back_to_harness_traceback(self, sample_settings):
        # dry_run skips launcher collection; only the harness traceback remains.
        launcher = MagicMock()
        harness = Harness(sample_settings, launcher, MagicMock(spec=LLMClient), dry_run=True)
        harness.step = 4
        harness.completion_reason = "crashed: boom"
        harness._crash_traceback = (
            'Traceback (most recent call last):\n'
            '  File "src/tester/harness.py", line 500, in analyze_and_act\n'
            'RuntimeError: boom\n'
        )
        harness._gather_final_error_context(exit_code=2)
        assert harness._error_source_file == "src/tester/harness.py"
        assert "RuntimeError: boom" in harness._error_excerpt

    def test_game_error_wins_over_harness_traceback_on_crash(self, sample_settings):
        launcher = MagicMock()
        launcher.get_stdout_tail.return_value = None
        launcher.get_stderr_tail.return_value = None
        launcher.read_engine_error_log.return_value = (
            'Traceback (most recent call last):\n'
            '  File "game/script.rpy", line 12, in script\n'
            'Exception: real game error\n'
        )
        harness = Harness(sample_settings, launcher, MagicMock(spec=LLMClient), dry_run=False)
        harness.completion_reason = "crashed: boom"
        harness._crash_traceback = (
            'Traceback (most recent call last):\n'
            '  File "src/tester/harness.py", line 1, in x\n'
            'RuntimeError: boom\n'
        )
        harness._gather_final_error_context(exit_code=2)
        # The game's own crash log, not the harness traceback, is surfaced.
        assert harness._error_source_file == "game/script.rpy"
        assert "real game error" in harness._error_excerpt


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


class TestCoordinateCalibration:
    """The startup probe should learn model-space → pixel transforms."""

    def test_fit_linear_recovers_scale(self):
        from tester.harness import Harness

        a, b = Harness._fit_linear([0.0, 100.0, 200.0], [0.0, 158.0, 316.0])
        assert abs(a - 1.58) < 1e-6
        assert abs(b) < 1e-6

    def test_fit_linear_degenerate(self):
        from tester.harness import Harness

        assert Harness._fit_linear([5.0, 5.0], [1.0, 2.0]) is None  # no x variance
        assert Harness._fit_linear([1.0], [1.0]) is None            # too few points

    def _calib_harness(self, sample_settings, image_size):
        from PIL import Image

        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(sample_settings, launcher, client, dry_run=True)
        harness.capturer = MagicMock()
        harness.capturer.capture_pil.return_value = Image.new("RGB", image_size)
        return harness, client

    def test_calibration_learns_normalized_transform(self, sample_settings):
        from tester.screen import Capturer

        W, H = 1578, 916
        harness, client = self._calib_harness(sample_settings, (W, H))
        _, markers = Capturer.draw_calibration_markers((W, H))
        # Model reports coordinates normalized to 0-1000 per axis (Qwen-VL style)
        client.locate_markers.return_value = {
            "markers": [
                {"label": lbl, "x": kx / W * 1000, "y": ky / H * 1000}
                for (lbl, kx, ky) in markers
            ]
        }

        harness._calibrate_coordinates()

        harness.capturer.set_coordinate_calibration.assert_called_once()
        (calib,), _ = harness.capturer.set_coordinate_calibration.call_args
        ax, bx, ay, by = calib
        assert abs(ax - W / 1000) < 0.01
        assert abs(ay - H / 1000) < 0.01
        assert abs(bx) < 1.0 and abs(by) < 1.0

    def test_calibration_falls_back_on_too_few_markers(self, sample_settings):
        harness, client = self._calib_harness(sample_settings, (1578, 916))
        client.locate_markers.return_value = {"markers": [{"label": 1, "x": 10, "y": 10}]}

        harness._calibrate_coordinates()

        harness.capturer.set_coordinate_calibration.assert_not_called()

    def test_calibration_swallows_probe_error(self, sample_settings):
        harness, client = self._calib_harness(sample_settings, (1578, 916))
        client.locate_markers.side_effect = RuntimeError("model offline")

        # Must not raise — falls back silently
        harness._calibrate_coordinates()
        harness.capturer.set_coordinate_calibration.assert_not_called()

    def test_calibration_rejects_poor_fit(self, sample_settings):
        from tester.screen import Capturer

        W, H = 1578, 916
        harness, client = self._calib_harness(sample_settings, (W, H))
        _, markers = Capturer.draw_calibration_markers((W, H))
        # Model reports near-identical coords for every marker → no real signal,
        # so any fit has large residuals and must be rejected.
        client.locate_markers.return_value = {
            "markers": [{"label": lbl, "x": 500 + lbl, "y": 500 + lbl} for (lbl, _, _) in markers]
        }

        harness._calibrate_coordinates()
        harness.capturer.set_coordinate_calibration.assert_not_called()
