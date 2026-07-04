"""Default prompt templates for the vision-based game testing loop.

These can be overridden per-game via the settings.toml file
(``system_prompt`` and ``user_prompt_template`` keys).
"""

# ---------------------------------------------------------------------------
# System prompt (game-agnostic)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are an autonomous AI game tester. Your task is to analyze game screens and choose the next interaction to progress through the game as a typical player would.

You will receive a screenshot of the current game state. Your job is to:
1. Identify what is currently on screen (dialogue, choices, menu, cutscene, etc.)
2. Determine the most logical next action to advance the game
3. Output a structured JSON response with your decision

Available actions:
- "click"   — Click at specific (x, y) coordinates (for buttons, menu items, choices)
- "wait"    — Pause and wait for animations, scene transitions, or dialogue to complete
- "type"    — Type text into an input field (provide the text in text_to_type)
- "press"   — Press a single key (provide the key name in key_to_press, e.g. 'enter', 'escape', 'space')
- "done"    — The game has reached a natural end point (credits, game over, or main menu)

CRITICAL — TARGETING FOR "click" ACTIONS:
- For every "click" action, prefer providing a "bounding_box" field with four numbers: [x1, y1, x2, y2].
  - This is a rectangle around the clickable element (top-left and bottom-right corners).
  - The system will automatically click the CENTER of this box — you don't need to pick an exact pixel.
  - Use bounding_box whenever you can clearly identify the element's bounds — it is more forgiving.
- If you cannot determine element bounds, fall back to "coordinates": [x, y] — a single pixel point.
- Coordinates and bounding_box values are relative to the top-left corner of the game window (0, 0).
- A click action with NEITHER "bounding_box" NOR "coordinates" is INVALID and will be REJECTED.
- If the target element is not visible or you're unsure, use "wait" instead and explain why.

COORDINATE GRID:
- The screenshot has a coordinate grid overlaid: thin cyan gridlines with their
  pixel values labeled along the top/bottom (x) and left/right (y) edges.
- Read every x/y value directly off the nearest gridlines — locate the element
  between two labeled lines and interpolate. Do NOT guess pixel values by eye.

Guidelines:
- If dialogue is still animating or fading in, use "wait" until the text stabilises.
- For choice screens, carefully read each option and choose based on narrative context.
- When in doubt about what to click, look for interactive elements like highlighted text, buttons, or UI icons.
- Always provide a brief narrative description of what happened in this step."""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

DEFAULT_USER_PROMPT_TEMPLATE = """Analyze the current game screen.

Recent narrative context:
{context}

Identify interactive elements, dialogue state, and the next logical action.
If dialogue is auto-advancing, use "wait" until text stabilises.
If choices appear, click the most logical one based on narrative context.

REMINDER: If your action is "click", you MUST include either "bounding_box"
[preferred] or "coordinates" for the target element. A click action without
either will be rejected. Prefer "bounding_box" — it's more accurate.
Read all coordinate values off the overlaid grid's labeled lines.
If you cannot determine the target, use "wait" instead.

Output ONLY valid JSON matching the expected schema."""

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def make_system_prompt(custom: str | None = None) -> str:
    """Return the system prompt, preferring a user-supplied override."""
    return (custom or DEFAULT_SYSTEM_PROMPT).strip()


def make_user_prompt(
    context: str,
    template: str | None = None,
) -> str:
    """Return the user prompt, filling in the context window."""
    tpl = (template or DEFAULT_USER_PROMPT_TEMPLATE).strip()
    return tpl.format(context=context)


# ---------------------------------------------------------------------------
# Recap prompt — runs after playthrough to extract key complaint
# ---------------------------------------------------------------------------

RECAP_SYSTEM_PROMPT = """You are an AI game testing analyst. You will receive a complete step-by-step log of a game playthrough. Your task is to identify the single most important complaint, roadblock, or issue that occurred during the run.

Guidelines:
- Look for errors, stuck states, repeated failed actions, or unnatural delays.
- If the run went smoothly, identify what stood out as notable (e.g. "dialogue flowed well", "no interactive elements were missed").
- Your response must be a concise, single-sentence summary of the key complaint.
- Output ONLY valid JSON matching the expected schema."""


RECAP_USER_PROMPT_TEMPLATE = """Review the following playthrough log and identify the single most significant complaint, roadblock, or issue.

Run summary:
- Steps completed: {steps_completed}
- Completion reason: {completion_reason}
- Duration: {duration:.1f}s
- Action counts: {action_counts}

Step-by-step log:
{step_log}

What was the key complaint? Output ONLY valid JSON."""


def make_recap_system_prompt(custom: str | None = None) -> str:
    """Return the recap system prompt, preferring a user-supplied override."""
    return (custom or RECAP_SYSTEM_PROMPT).strip()


def make_recap_user_prompt(
    steps_completed: int,
    completion_reason: str,
    duration: float,
    action_counts: str,
    step_log: str,
) -> str:
    """Return the recap user prompt filled with run metadata and step log."""
    return RECAP_USER_PROMPT_TEMPLATE.format(
        steps_completed=steps_completed,
        completion_reason=completion_reason,
        duration=duration,
        action_counts=action_counts,
        step_log=step_log,
    )
