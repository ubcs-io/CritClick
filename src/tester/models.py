"""Pydantic models for LLM-structured output and internal data types."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Valid actions the harness can execute."""

    CLICK = "click"
    WAIT = "wait"
    TYPE = "type"
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
        description="(x, y) pixel coordinates relative to top-left of the game window. "
        "Required for 'click' actions. May be empty for 'wait'/'done'.",
    )
    text_to_type: Optional[str] = Field(
        None,
        description="Text to input for 'type' actions. Ignored otherwise.",
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


class LogEntry(BaseModel):
    """A single step entry written to the structured log file."""

    step: int
    timestamp: str
    llm_response: ActionResponse
    image_hex_truncated: str = Field(
        ..., description="First 200 chars of the base64-encoded screenshot for traceability."
    )
    screenshot_path: Optional[str] = None


class GameConfig(BaseModel):
    """Runtime configuration for a specific game under test."""

    type: str = Field(
        ..., description="Engine type: 'renpy', 'godot', or 'custom'."
    )
    path: str = Field(
        ..., description="Path to the game directory or executable."
    )
    executable: Optional[str] = Field(
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
