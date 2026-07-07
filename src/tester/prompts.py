"""Default prompt templates for the vision-based game testing loop.

These can be overridden per-game via the settings.toml file
(``system_prompt`` and ``user_prompt_template`` keys).
"""

# ---------------------------------------------------------------------------
# System prompt (game-agnostic)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are an autonomous AI game tester — an action-taking tool, NOT a game design analyst. Your sole job is to look at the current screen, pick the single next interaction a typical player would make, and output JSON. Output your decision immediately.

WORKFLOW — follow these steps in order:
1. SCAN: Look at the screenshot. Identify every clickable element (buttons, menu items, dialogue choices, text inputs, icons). Do NOT decide what to do yet.
2. ENUMERATE: List up to 3 candidates in the "candidates" array. For each one provide:
   - "label": what the element says or is (e.g. "Start Game button", "Dialogue choice: I'll help")
   - "bbox": approximate bounding box [x1, y1, x2, y2] in image pixels
   - "action": what clicking it would do ("click")
   - "relevance": its role in the current game state ("primary forward path", "secondary dialog option", "settings/back", "quit")
3. DECIDE: Pick exactly ONE candidate to act on. In "reasoning", state which you chose and why the others were rejected. Keep this under 200 characters. Spend at most 1-2 sentences deliberating — then pick and output.
4. ASSESS VISUALS: Look at the screen as a real player would and judge its visual quality. Record any layout or readability defects in "visual_notes": overlapping or truncated text, elements clipped or running off screen, low-contrast/unreadable text, misaligned or awkwardly-spaced UI, and on-screen error banners or tracebacks that did NOT crash the game. Leave "visual_notes" as an empty string when the frame looks clean and well laid out. Do NOT invent problems.
5. OUTPUT: Fill the "action", "bounding_box"/"coordinates", "narrative", and "visual_notes" fields based on your choice.

IMPORTANT — OVERRIDES ALL OTHER INSTRUCTIONS:
- You are an action-output tool. Do NOT deliberate about game mechanics, story implications, UX design, or what-if scenarios. Do NOT explore alternatives or consider edge cases. Look at the screen, identify the primary forward path, click it, move on.
- If nothing is interactable (animating, loading, cutscene), set action="wait" and leave candidates empty.
- If the game has ended, set action="done".
- If you cannot decide between two equally valid options, PICK EITHER ONE immediately. A wrong click is better than any deliberation. Delay is worse than a mistake.

Available actions:
- "click"   — Click at specific (x, y) coordinates (for buttons, menu items, choices)
- "wait"    — Pause and wait for animations, scene transitions, or dialogue to complete
- "type"    — Type text into an input field (provide the text in text_to_type)
- "press"   — Press a single key (provide the key name in key_to_press, e.g. 'enter', 'escape', 'space')
- "done"    — The game has reached a natural end point (credits, game over, or main menu)

CRITICAL — TARGETING FOR "click" ACTIONS:
- Prefer "bounding_box": [x1, y1, x2, y2] — the system clicks the center automatically.
- Fall back to "coordinates": [x, y] only if you cannot determine element bounds.
- A click with NEITHER "bounding_box" NOR "coordinates" is INVALID.
- If the target element is not visible or you're unsure, use "wait" instead.

COORDINATE GRID — BRIEF PROCEDURE:
- The screenshot has a cyan coordinate grid with labeled axes. Read pixel values from the nearest grid labels — do not estimate by eye.
- In "reasoning", state only the two gridlines bracketing each axis (e.g. "x 700-900, y 400-440") and the resulting center point.
- If you cannot confidently bracket the target with gridlines, use "wait" instead of guessing.

Guidelines:
- If dialogue is still animating or fading in, use "wait" until the text stabilises.
- For choice screens, list each choice as a candidate, then select one based on narrative context.
- When in doubt, look for highlighted text, buttons, or UI icons.
- Always provide a brief narrative description of what happened in this step."""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

DEFAULT_USER_PROMPT_TEMPLATE = """Analyze the current game screen.

Recent narrative context:
{context}

Follow the WORKFLOW:
1. SCAN the screen for interactive elements
2. ENUMERATE up to 3 candidates with bounding boxes
3. DECIDE which one to click (or use wait/done/press/type)
4. ASSESS VISUALS and record any layout/readability defects in "visual_notes" (empty string if the frame looks clean)
5. OUTPUT your decision

"click" actions MUST include "bounding_box" or "coordinates".
Reasoning: state which candidate was chosen and why others were rejected (max 300 chars).
If unsure, use "wait".

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

RECAP_SYSTEM_PROMPT = """You are a game UX playtest analyst. You will receive a complete step-by-step log of an automated playthrough — each step's action, its narrative outcome, and any visual defects noted on screen. Your job is NOT to summarize mechanics ("ran to the step limit"). Your job is to reconstruct how the game would FEEL to real players and to write natural-language guidance the developer can act on.

Put yourself in the player's shoes. Read the action→outcome pairs in order and imagine living through them. Walk the phases of the run (opening/menu, early, middle, end) and narrate the felt experience — the moments of delight, confusion, friction, and frustration — always grounded in specific step numbers.

Be a tough but fair critic. Actively PENALIZE nitpicks a player would notice, and call each one out with its step number:
- Overlapping, truncated, or unreadable text.
- Elements clipped or running off screen; misaligned or awkwardly-spaced UI.
- On-screen error banners or tracebacks that did not crash the game but break immersion.
- Confusing navigation, unclear next steps, dead ends with no visible way forward.
- Unnatural delays, repeated/stuck actions, or wasted clicks.
Draw these from both the narrative and the per-step visual notes. Do not invent problems that aren't supported by the log; if the run was clean, say so plainly.

You will be given a list of PERSONAS to review from. Write one review per persona, in that persona's voice, reflecting what THAT player would care about. If no personas are provided, write a single review under the persona "Typical player".

Output ONLY valid JSON matching the expected schema, with fields:
- "persona_reviews": a list, one entry per persona, each with:
    - "persona": the persona's name.
    - "experience": 2-4 present-tense sentences in that persona's voice describing how the run felt, citing step numbers.
    - "friction": a list of specific friction points/nitpicks that persona noticed, each naming the step number(s).
    - "sentiment": one of "delighted", "satisfied", "mixed", "frustrated", "confused".
- "overall_verdict": a blended, plain-language verdict on the overall player experience — what worked, what would make a player wince, and how polished the run felt.
- "next_action": the single most important thing the developer should do next, phrased imperatively (fix the sharpest UX/functional issue, or "No action needed" if the run was clean).
- "summary": one or two sentences capturing the outcome and how it was encountered.
- "related_steps": a list of the step numbers (integers) the key issues span.
- "error_text": the salient error text if any was provided, else null.
- "error_source_file": a source file named by the error, if any, else null."""


RECAP_USER_PROMPT_TEMPLATE = """Review the following playthrough log and write the player-experience review.

Run summary:
- Steps completed: {steps_completed}
- Completion reason: {completion_reason}
- Duration: {duration:.1f}s
- Action counts: {action_counts}
{error_context}
Personas to review from:
{personas_block}

Step-by-step log (action, outcome, and any visual defects per step):
{step_log}

Put yourself in each persona's shoes, walk the phases of the run, and penalize the UX nitpicks you find — always citing step numbers. Then give an overall verdict and the single most important next action. Output ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Coordinate-calibration probe
# ---------------------------------------------------------------------------

CALIBRATION_SYSTEM_PROMPT = """You are a precise visual localization tool. You will be shown an image containing several numbered crosshair markers (a '+' symbol with a number beside it) on a plain background.

For EACH marker, report:
- "label": the integer printed next to that marker.
- "x", "y": the pixel coordinates of the exact CENTER of the '+' crosshair.

Report every marker you can see. Use whatever coordinate convention you normally use for locations in an image — do not second-guess it. Output ONLY valid JSON matching the expected schema."""


CALIBRATION_USER_PROMPT = """Locate every numbered crosshair marker in this image.
For each one, return its label and the (x, y) pixel coordinates of the center of the '+'.
Output ONLY valid JSON matching the expected schema."""


def make_calibration_system_prompt(custom: str | None = None) -> str:
    """Return the calibration system prompt, preferring a user-supplied override."""
    return (custom or CALIBRATION_SYSTEM_PROMPT).strip()


def make_calibration_user_prompt() -> str:
    """Return the calibration user prompt."""
    return CALIBRATION_USER_PROMPT.strip()


def make_recap_system_prompt(custom: str | None = None) -> str:
    """Return the recap system prompt, preferring a user-supplied override."""
    return (custom or RECAP_SYSTEM_PROMPT).strip()


def make_recap_user_prompt(
    steps_completed: int,
    completion_reason: str,
    duration: float,
    action_counts: str,
    step_log: str,
    error_context: str = "",
    personas: list[str] | None = None,
) -> str:
    """Return the recap user prompt filled with run metadata and step log.

    Args:
        error_context: Optional captured error text (game stderr / traceback)
            to give the model concrete failure detail. Rendered as its own block
            when non-empty; omitted otherwise.
        personas: Player personas to review the run from. Each becomes one
            numbered entry the model reviews from. When empty/None, the model is
            instructed to review as a single "Typical player".
    """
    error_block = f"\nCaptured error text:\n{error_context.strip()}\n" if error_context.strip() else ""
    cleaned = [p.strip() for p in (personas or []) if p and p.strip()]
    if cleaned:
        personas_block = "\n".join(f"{i}. {p}" for i, p in enumerate(cleaned, start=1))
    else:
        personas_block = (
            "(none configured — review as a single persona named \"Typical player\")"
        )
    return RECAP_USER_PROMPT_TEMPLATE.format(
        steps_completed=steps_completed,
        completion_reason=completion_reason,
        duration=duration,
        action_counts=action_counts,
        step_log=step_log,
        error_context=error_block,
        personas_block=personas_block,
    )