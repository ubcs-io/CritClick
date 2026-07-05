"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from tester.models import (
    ActionResponse,
    ActionType,
    LogEntry,
    NextActionReport,
    NextActionStatus,
    RecapReport,
)


class TestActionType:
    def test_values(self):
        assert ActionType.CLICK.value == "click"
        assert ActionType.WAIT.value == "wait"
        assert ActionType.TYPE.value == "type"
        assert ActionType.PRESS.value == "press"
        assert ActionType.DONE.value == "done"

    def test_from_string(self):
        assert ActionType("click") is ActionType.CLICK
        assert ActionType("press") is ActionType.PRESS


class TestActionResponse:
    def test_valid_click(self):
        resp = ActionResponse(
            description="A button",
            action="click",
            coordinates=[100.0, 200.0],
            reasoning="Need to click button",
            narrative="Clicked button",
        )
        assert resp.action is ActionType.CLICK
        assert resp.coordinates == [100.0, 200.0]
        assert resp.text_to_type is None
        assert resp.key_to_press is None

    def test_valid_press(self):
        resp = ActionResponse(
            description="Dialogue",
            action="press",
            key_to_press="enter",
            reasoning="Advance dialogue",
            narrative="Pressed enter",
        )
        assert resp.action is ActionType.PRESS
        assert resp.key_to_press == "enter"

    def test_valid_type(self):
        resp = ActionResponse(
            description="Input field",
            action="type",
            text_to_type="hello",
            reasoning="Need to type",
            narrative="Typed text",
        )
        assert resp.action is ActionType.TYPE
        assert resp.text_to_type == "hello"

    def test_valid_done(self):
        resp = ActionResponse(
            description="Credits",
            action="done",
            reasoning="Game over",
            narrative="Game ended",
        )
        assert resp.action is ActionType.DONE

    def test_coordinates_default_empty(self):
        resp = ActionResponse(
            description="x",
            action="wait",
            reasoning="x",
            narrative="x",
        )
        assert resp.coordinates == []

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ActionResponse(action="click")  # type: ignore[call-arg]


class TestLogEntry:
    def test_with_duration(self):
        resp = ActionResponse(
            description="x",
            action="wait",
            reasoning="x",
            narrative="x",
        )
        entry = LogEntry(
            step=1,
            timestamp="2024-01-01T00:00:00",
            llm_response=resp,
            duration_ms=1500,
            image_hex_truncated="abc123",
        )
        assert entry.duration_ms == 1500

    def test_duration_default_zero(self):
        resp = ActionResponse(
            description="x",
            action="wait",
            reasoning="x",
            narrative="x",
        )
        entry = LogEntry(
            step=1,
            timestamp="2024-01-01T00:00:00",
            llm_response=resp,
            image_hex_truncated="abc123",
        )
        assert entry.duration_ms == 0

    def test_duration_negative_rejected(self):
        resp = ActionResponse(
            description="x",
            action="wait",
            reasoning="x",
            narrative="x",
        )
        with pytest.raises(ValidationError):
            LogEntry(
                step=1,
                timestamp="2024-01-01T00:00:00",
                llm_response=resp,
                duration_ms=-1,
                image_hex_truncated="abc123",
            )


class TestNextActionReport:
    def test_valid_report(self):
        report = NextActionReport(
            next_action="Investigate game/script.rpy — the game crashed at step 7.",
            summary="The game process died unexpectedly at step 7.",
            related_steps=[7],
            status=NextActionStatus.GAME_DIED,
            severity="error",
        )
        assert report.related_steps == [7]
        assert report.status == NextActionStatus.GAME_DIED

    def test_defaults(self):
        report = NextActionReport(next_action="No action needed.")
        assert report.summary == ""
        assert report.error_text is None
        assert report.related_steps == []
        assert report.status == NextActionStatus.UNKNOWN
        assert report.severity == "info"

    def test_missing_next_action(self):
        with pytest.raises(ValidationError):
            NextActionReport()

    def test_key_complaint_back_compat(self):
        # key_complaint falls back to summary, then next_action.
        report = NextActionReport(next_action="Do X.", summary="Y happened.")
        assert report.key_complaint == "Y happened."
        report_no_summary = NextActionReport(next_action="Do X.")
        assert report_no_summary.key_complaint == "Do X."

    def test_recap_report_is_alias(self):
        # RecapReport remains importable as a NextActionReport subclass.
        assert issubclass(RecapReport, NextActionReport)
        report = RecapReport(next_action="Do X.")
        assert isinstance(report, NextActionReport)
