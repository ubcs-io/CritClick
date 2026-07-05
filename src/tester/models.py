"""Pydantic models for LLM-structured output and internal data types."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Valid actions the harness can execute."""

    CLICK = "click"
    WAIT = "wait"
    TYPE = "type"
    PRESS = "press"
    DONE = "done"


class ActionResponse(BaseModel):
    """Structured response expected from the vision LLM."""

    description: str = Field(
        ..., description="Brief description of what's on screen."
    )
    action: ActionType = Field(
        ...,
        description=(
            "The action to take: 'click' to tap UI, 'wait' to pause for animation, "
            "'type' to input text, 'done' when the game has ended or returned to menu."
        ),
    )
    coordinates: list[float] = Field(
        default_factory=list,
        description="(x, y) pixel coordinates relative to top-left of the game window "
        "for 'click' actions. Use bounding_box instead for more precise targeting. "
        "May be empty for 'wait'/'done'.",
    )
    bounding_box: list[float] | None = Field(
        None,
        description="[x1, y1, x2, y2] bounding box of the target element in pixels "
        "relative to top-left of the game window. When provided for 'click' actions, "
        "the click will be centered on this box. Prefer this over raw coordinates "
        "for better targeting precision.",
    )
    text_to_type: str | None = Field(
        None,
        description="Text to input for 'type' actions. Ignored otherwise.",
    )
    key_to_press: str | None = Field(
        None,
        description=(
            "Single key to press for 'press' actions (e.g. 'enter', 'escape', 'space'). "
            "Ignored otherwise."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Chain-of-thought explanation for why this action was chosen.",
    )
    narrative: str = Field(
        ...,
        description="In-character or descriptive narrative of what happened this step, "
        "written in present tense (e.g. '💬 Character greets the player.').",
    )


class MarkerLocation(BaseModel):
    """One calibration marker located by the vision model."""

    label: int = Field(..., description="The number printed next to the marker.")
    x: float = Field(..., description="X coordinate of the marker's center, in the model's coordinate space.")
    y: float = Field(..., description="Y coordinate of the marker's center, in the model's coordinate space.")


class CalibrationResponse(BaseModel):
    """Model response for the coordinate-calibration probe image."""

    markers: list[MarkerLocation] = Field(
        default_factory=list,
        description="One entry per numbered crosshair marker visible in the image.",
    )


class NextActionStatus(str, Enum):
    """Classification of how a run ended, driving the next-action takeaway."""

    GAME_DIED = "game_died"
    CRASHED = "crashed"
    CONFIG_ERROR = "config_error"
    STUCK = "stuck"
    MAX_STEPS = "max_steps"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class NextActionReport(BaseModel):
    """The single salient takeaway from a run: the next key action to take.

    Rendered identically into ``NEXT_ACTION.md``, the run manifest, and the
    stdout/stderr summaries so there is one source of truth per run.
    """

    next_action: str = Field(
        ...,
        description=(
            "The single most important next action to take, phrased imperatively "
            "(e.g. 'Investigate game/script.rpy — the game crashed at step 7')."
        ),
    )
    summary: str = Field(
        default="",
        description=(
            "Descriptive summary of the problem or outcome and the way it was "
            "encountered."
        ),
    )
    error_text: str | None = Field(
        default=None,
        description="Specific error text captured from the run, verbatim, if any.",
    )
    error_source_file: str | None = Field(
        default=None,
        description="Source file referenced by the error, if the error names one.",
    )
    related_steps: list[int] = Field(
        default_factory=list,
        description="Playthrough step numbers the issue spans or was encountered at.",
    )
    status: NextActionStatus = Field(
        default=NextActionStatus.UNKNOWN,
        description="How the run ended, classifying the takeaway.",
    )
    severity: str = Field(
        default="info",
        description="Severity of the takeaway: 'error', 'warning', or 'info'.",
    )

    @property
    def key_complaint(self) -> str:
        """Back-compat accessor: older readers expect a single complaint string."""
        return self.summary or self.next_action


class RecapReport(NextActionReport):
    """Backwards-compatible alias for :class:`NextActionReport`.

    Retained so existing code/manifests referring to ``RecapReport`` and its
    ``key_complaint`` field keep working.
    """


class LogEntry(BaseModel):
    """A single step entry written to the structured log file."""

    step: int
    timestamp: str
    llm_response: ActionResponse
    duration_ms: int = Field(
        default=0, ge=0, description="Milliseconds spent on the LLM call for this step."
    )
    image_hex_truncated: str = Field(
        ..., description="First 200 chars of the base64-encoded screenshot for traceability."
    )
    screenshot_path: str | None = None


class GameConfig(BaseModel):
    """Runtime configuration for a specific game under test."""

    type: str = Field(
        ..., description="Engine type: 'renpy', 'godot', or 'custom'."
    )
    path: str = Field(
        ..., description="Path to the game directory or executable."
    )
    executable: str | None = Field(
        None,
        description=(
            "Override executable name/path. For Ren'Py: 'renpy.sh'/'renpy.exe'. "
            "For Godot: path to the Godot binary. For custom: full executable path."
        ),
    )
    resolution: tuple[int, int] = Field(
        default=(1280, 720),
        description="Desired game window resolution as (width, height).",
    )
    git_pull: bool = Field(
        default=False,
        description="If True, run 'git pull --rebase' before launching.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Additional CLI arguments to pass to the game executable.",
    )
