"""Core vision-based playthrough harness.

Orchestrates the capture → analyse → act → log loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pyautogui
from PIL import Image

from .client import LLMClient
from .config import Settings
from .launcher import Launcher, capture_git_info
from .models import (
    ActionResponse,
    ActionType,
    CalibrationResponse,
    LogEntry,
    NextActionReport,
    NextActionStatus,
    SaveState,
)
from .perception import classify_change, frame_delta, frame_signature
from .prompts import (
    make_calibration_system_prompt,
    make_calibration_user_prompt,
    make_recap_system_prompt,
    make_recap_user_prompt,
    make_system_prompt,
    make_user_prompt,
)
from .screen import Capturer, ScreenCaptureError
from .window import WindowTracker

logger = logging.getLogger("tester.harness")

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_GAME_DIED = 3
EXIT_INTERRUPTED = 130
EXIT_LOCKED = 4


class Harness:
    """Vision-based playthrough harness.

    The main loop:
      1. Capture screen → base64
      2. Send to LLM → structured JSON
      3. Validate & execute action (click / wait / type / press / done)
      4. Log everything to JSONL

    Usage::

        settings = Settings.from_toml("my_game.toml")
        launcher = create_launcher(settings.game)
        client = OpenAIClient(...)
        harness = Harness(settings, launcher, client)
        harness.run()
    """

    def __init__(
        self,
        settings: Settings,
        launcher: Launcher,
        llm_client: LLMClient,
        dry_run: bool = False,
        run_id: str | None = None,
        runs_dir: str | Path = "runs",
        debug_harness: bool = False,
        debug_screen: bool = False,
        save_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.launcher = launcher
        self.llm = llm_client
        self.dry_run = dry_run
        self.debug_harness = debug_harness
        self.debug_screen = debug_screen
        self.save_id = save_id
        self.capturer = Capturer(
            scale=settings.harness.screen_scale,
            debug=debug_screen,
            game_resolution=tuple(settings.game.resolution),
        )
        # Config fallback for the model's coordinate convention (used when
        # calibration is disabled or fails).
        self.capturer.set_coordinate_max(settings.harness.coordinate_max)

        # Run namespacing — every run gets a unique ID so outputs are always
        # isolated under ``<runs_dir>/<run_id>/``.  If the caller doesn't
        # supply one, a default timestamp-based ID is generated.
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)

        # Runtime state
        self.step = 0
        self.step_history: list[str] = []
        self.stuck_counter = 0
        self.last_action: str | None = None
        self.start_time: float | None = None
        self.completion_reason: str | None = None
        self.action_counts: dict[str, int] = {}
        self._recovery_attempts = 0
        self._recovery_strategies: list[str] = []
        self._window_tracker: WindowTracker | None = None
        # Count of advance-macro frames saved this step (for unique filenames).
        self._macro_frame_seq = 0

        # Save/resume state — tracks cumulative progress across threaded runs.
        self._total_steps_across_runs: int = 0
        self._save_triggered: bool = False

        # Adaptive reasoning throttling — tightens max_tokens per-call when
        # the model burns budget on thinking without producing content.
        self._effective_max_tokens: int = settings.llm.max_tokens
        self._reasoning_penalties: int = 0
        # Step-down sequence: each penalty shrinks the budget by one tier.
        # Once at the floor, no further tightening.
        _BUDGET_TIERS = (600, 450, 300, 200)

        # Captured at run start, written into the manifest
        self._git_info: dict | None = None
        self._exit_code: int | None = None

        # Error context captured during the run, used to build the next-action
        # takeaway (game engine tracebacks, harness crash tracebacks, and any
        # source file / excerpt extracted from them).
        self._game_stdout_tail: str | None = None
        self._game_stderr_tail: str | None = None
        self._crash_traceback: str | None = None
        self._error_source_file: str | None = None
        self._error_excerpt: str | None = None

        # Apply namespacing to settings in-place before setting up output.
        if run_id:
            self._apply_run_namespace()

        # Setup output directories
        self._setup_output()

        if not dry_run:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1

        logger.info(
            "🎮 Harness initialised | max_steps=%d | model=%s | endpoint=%s | dry_run=%s",
            settings.harness.max_steps,
            settings.llm.model,
            settings.llm.api_base,
            dry_run,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _apply_run_namespace(self) -> None:
        """Rewrite logging paths to live under ``<runs_dir>/<run_id>/``.

        Mutates ``self.settings.logging`` in place so the rest of the harness
        reads/writes the namespaced paths transparently.
        """
        run_dir = self.runs_dir / self.run_id  # type: ignore[arg-type]
        lg = self.settings.logging
        lg.log_file = str(run_dir / "playthrough_log.jsonl")
        lg.screenshot_dir = str(run_dir / "screenshots")

    def _setup_output(self) -> None:
        """Create screenshot dir and initialise the log file for a fresh run."""
        log_settings = self.settings.logging
        if log_settings.save_screenshots:
            os.makedirs(log_settings.screenshot_dir, exist_ok=True)
        # Truncate the log file so the recap only sees this run's entries.
        # Prior runs were written to the same path in legacy mode (no run_id)
        # or when a run_id is reused; keeping them would pollute the summary.
        Path(log_settings.log_file).parent.mkdir(parents=True, exist_ok=True)
        open(log_settings.log_file, "w", encoding="utf-8").close()

    # ------------------------------------------------------------------
    # Coordinate calibration
    # ------------------------------------------------------------------

    def _calibrate_coordinates(self) -> None:
        """Learn how the model's coordinates map to pixels via a probe image.

        Renders a probe with numbered markers at known pixel positions, asks
        the model to locate them, and fits a per-axis affine transform
        (``pixel = a·model + b``). On success the transform is applied to all
        clicks; on failure we keep the ``coordinate_max`` fallback (or identity).
        """
        try:
            import base64

            # Probe must match the real capture size — a model's coordinate
            # convention can depend on the input image dimensions.
            probe_size = self.capturer.capture_pil().size
            probe_w, probe_h = probe_size
            probe_img, markers = Capturer.draw_calibration_markers(probe_size)

            buf = BytesIO()
            probe_img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            raw = self.llm.locate_markers(
                image_b64,
                make_calibration_system_prompt(),
                make_calibration_user_prompt(),
            )
            parsed = CalibrationResponse.model_validate(raw)

            # Match reported markers to known positions by label.
            known = {label: (x, y) for label, x, y in markers}
            model_x: list[float] = []
            pixel_x: list[float] = []
            model_y: list[float] = []
            pixel_y: list[float] = []
            for m in parsed.markers:
                if m.label in known:
                    kx, ky = known[m.label]
                    model_x.append(m.x)
                    pixel_x.append(kx)
                    model_y.append(m.y)
                    pixel_y.append(ky)

            if len(model_x) < 2:
                logger.warning(
                    "🧭 Calibration: model returned %d usable markers (<2) — "
                    "keeping fallback (coordinate_max=%s).",
                    len(model_x), self.settings.harness.coordinate_max,
                )
                return

            fit_x = self._fit_linear(model_x, pixel_x)
            fit_y = self._fit_linear(model_y, pixel_y)
            if fit_x is None or fit_y is None:
                logger.warning("🧭 Calibration: degenerate fit — keeping fallback.")
                return

            ax, bx = fit_x
            ay, by = fit_y
            if ax <= 0.01 or ay <= 0.01:
                logger.warning(
                    "🧭 Calibration: implausible scale (a_x=%.4f, a_y=%.4f) — "
                    "keeping fallback.", ax, ay,
                )
                return

            # Reject fits where the model localized markers poorly: the largest
            # residual (fitted vs known pixel) must stay within a fraction of
            # the image, otherwise the transform is untrustworthy.
            max_res = 0.0
            for mx, kx in zip(model_x, pixel_x):
                max_res = max(max_res, abs(ax * mx + bx - kx))
            for my, ky in zip(model_y, pixel_y):
                max_res = max(max_res, abs(ay * my + by - ky))
            tol = 0.1 * max(probe_w, probe_h)
            if max_res > tol:
                logger.warning(
                    "🧭 Calibration: fit residual %.1fpx exceeds tolerance %.1fpx "
                    "(model localized markers poorly) — keeping fallback.",
                    max_res, tol,
                )
                return

            self.capturer.set_coordinate_calibration((ax, bx, ay, by))
            logger.info(
                "🧭 Coordinate calibration learned from %d/%d markers: "
                "x_px=%.4f·x%+.1f | y_px=%.4f·y%+.1f",
                len(model_x), len(markers), ax, bx, ay, by,
            )
            if self.debug_screen:
                for m in parsed.markers:
                    if m.label in known:
                        kx, ky = known[m.label]
                        fx, fy = ax * m.x + bx, ay * m.y + by
                        logger.info(
                            "🧭 marker %d: model=(%.1f, %.1f) → fitted=(%.1f, %.1f) "
                            "| known=(%d, %d) | residual=(%.1f, %.1f)",
                            m.label, m.x, m.y, fx, fy, kx, ky, fx - kx, fy - ky,
                        )

        except Exception as exc:
            logger.warning("🧭 Calibration probe failed: %s — keeping fallback.", exc)

    @staticmethod
    def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
        """Least-squares fit ``y = a·x + b``. Returns ``(a, b)`` or None if degenerate."""
        n = len(xs)
        if n < 2:
            return None
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            return None
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
        return (a, b)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run the full playthrough loop.

        Returns:
            Exit code (0=success, 2=runtime error, 3=game died, 130=interrupted).
        """
        logger.info("🎮 Starting autonomous vision-based playthrough…")
        self.start_time = time.time()

        # If a save_id was provided, try to resume from a previous session.
        # This must happen before the git-info capture so the manifest
        # reflects this run's start state correctly.
        if self.save_id:
            self._load_save_state()
            logger.info(
                "💾 Save/resume active | save_id=%s | starting at step %d",
                self.save_id, self.step + 1,
            )

        # Capture git state of the game repo at run start for the manifest.
        # Done before pull_latest() so the manifest reflects what was tested.
        try:
            self._git_info = capture_git_info(self.settings.game.path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"⚠️  Could not capture git info: {exc}")
            self._git_info = {"commit": None, "branch": None, "dirty": None}

        if self.dry_run:
            logger.info("🔍 DRY RUN MODE — no actions will be executed, game will not launch.")
        else:
            # Optional git pull
            self.launcher.pull_latest()

            # Startup countdown for safety
            if self.settings.harness.startup_countdown:
                self._startup_countdown()

            # Launch the game
            self.launcher.launch()

            # Find and bind to the game window for cropped capture + offset clicks
            pid = self.launcher.get_pid()
            if pid is not None:
                self._window_tracker = WindowTracker(
                    pid=pid,
                    timeout=self.settings.harness.window_find_timeout,
                    debug=self.debug_harness,
                )
                win = self._window_tracker.find_window()
                if win is not None:
                    self.capturer.set_game_window(win, self._window_tracker)
                    # Focus the game window once before the loop begins
                    self._window_tracker.focus(win)
                    time.sleep(0.3)
                else:
                    logger.info(
                        "🪟 No game window found — using full-screen capture + absolute coords."
                    )
            else:
                logger.info(
                    "🪟 Could not determine game PID — using full-screen capture + absolute coords."
                )

            # Learn how the model's coordinates map to pixels before the loop.
            if self.settings.harness.coordinate_calibration:
                self._calibrate_coordinates()

        exit_code = EXIT_SUCCESS
        try:
            for step_num in range(1, self.settings.harness.max_steps + 1):
                self.step = step_num
                if not self.dry_run and not self.launcher.is_alive():
                    logger.warning("⚠️  Game process died unexpectedly.")
                    self.completion_reason = "game_died"
                    exit_code = EXIT_GAME_DIED
                    break

                result = self.analyze_and_act()
                if result == "completed":
                    logger.info("🏁 Playthrough completed naturally.")
                    self.completion_reason = "completed"
                    break

                # When the save-trigger step has executed, stop the loop cleanly.
                # The harness state will be serialised in the finally block.
                if self._save_triggered:
                    logger.info(
                        "💾 Save step executed at step %d — stopping run cleanly. "
                        "State will be saved for the next run with --save-id=%s.",
                        self.step, self.save_id,
                    )
                    self.completion_reason = "saved"
                    break

                time.sleep(self.settings.harness.wait_after_action)
            else:
                self.completion_reason = "max_steps_reached"
                logger.info("📋 Reached max_steps (%d) — stopping.", self.settings.harness.max_steps)

        except KeyboardInterrupt:
            logger.info("⌨️ Playthrough interrupted by user.")
            self.completion_reason = "interrupted"
            exit_code = EXIT_INTERRUPTED
        except Exception as exc:
            logger.exception(f"💥 Harness crashed: {exc}")
            self.completion_reason = f"crashed: {exc}"
            exit_code = EXIT_RUNTIME_ERROR
            # Preserve the harness traceback. Game-side error context (the
            # engine crash log + the continuously-drained stderr) is gathered in
            # the finally block below and preferred over this when present, so
            # the takeaway cites the *game's* error rather than ours.
            self._crash_traceback = traceback.format_exc()
        finally:
            self.stop()
            self._exit_code = exit_code
            self._gather_final_error_context(exit_code)
            report = self._build_next_action()
            self._log_summary(report)
            self._write_next_action(report)
            self._write_manifest(next_action=report)
            # Write the save-state file if this was a save-enabled session.
            # The state captures the last completed step so the next run
            # resumes from step+1.
            if self.save_id and self.step > 0:
                self._write_save_state()

        return exit_code

    def stop(self) -> None:
        """Clean shutdown — stop the game, finalise logging."""
        if not self.dry_run:
            self.launcher.stop()
        logger.info("🛑 Harness stopped.")

    # Recognisable markers of an actual error/traceback in captured output.
    # Used to avoid mistaking benign engine warnings on a clean run for a
    # failure. Covers Python/Ren'Py tracebacks and Godot error lines.
    _ERROR_SIGNAL_RE = re.compile(
        r"Traceback \(most recent call last\)|"
        r"Exception:|"
        r"\bError:|"
        r"SCRIPT ERROR|USER ERROR|\bERROR:|"
        r"Segmentation fault|Fatal error|"
        r'File "[^"]+", line \d+',
        re.MULTILINE,
    )

    def _looks_like_error(self, text: str | None) -> bool:
        """Whether *text* contains a recognisable error/traceback signature."""
        return bool(text and self._ERROR_SIGNAL_RE.search(text))

    def _gather_final_error_context(self, exit_code: int) -> None:
        """Collect game-side error context once, after the game has stopped.

        Runs for every termination mode. Hard failures (game death, crash,
        config/runtime error) surface whatever the game last emitted; soft
        outcomes only surface output that actually looks like an error, so a
        clean run isn't reported as failing on benign engine warnings. Finally,
        a harness crash with no game-side error falls back to our own traceback.
        """
        reason = self.completion_reason or ""
        is_hard = (
            reason == "game_died"
            or reason.startswith("crashed:")
            or exit_code in (EXIT_CONFIG_ERROR, EXIT_RUNTIME_ERROR)
        )
        if not self.dry_run:
            self._collect_error_context(require_error_signal=not is_hard)

        # Harness-crash fallback: if the game itself surfaced nothing, mine the
        # harness traceback so we still name a source file + excerpt.
        if self._crash_traceback and not self._error_source_file and not self._error_excerpt:
            self._error_source_file, self._error_excerpt = self._extract_error_source(
                self._crash_traceback
            )

    def _collect_error_context(self, require_error_signal: bool = False) -> None:
        """Pull the game's stdout/stderr tails and engine crash log, non-blocking.

        Reads from the launcher's continuously-drained buffers (safe whether or
        not the game is still alive) plus the engine's own crash log file. Sets
        ``_game_stdout_tail``/``_game_stderr_tail`` for visibility and, when an
        error is found, ``_error_source_file``/``_error_excerpt``.

        Args:
            require_error_signal: When True, stderr/stdout are only mined for an
                error when they actually look like one (see ``_looks_like_error``).
                The engine crash log is always trusted — it only exists on a real
                crash.
        """
        try:
            stdout_tail = self.launcher.get_stdout_tail()
            stderr_tail = self.launcher.get_stderr_tail()
            engine_log = self.launcher.read_engine_error_log()
        except Exception as exc:  # defensive: never let this break shutdown
            logger.debug("Could not collect game error context: %s", exc)
            return

        if stdout_tail:
            self._game_stdout_tail = stdout_tail
        if stderr_tail:
            self._game_stderr_tail = stderr_tail
        if engine_log:
            logger.info("🧾 Engine crash log:\n%s", engine_log)
        elif stderr_tail:
            logger.info("📥 Game stderr:\n%s", stderr_tail)

        # Choose which stream to mine for a source file + excerpt. The engine's
        # own crash log is always a genuine error and wins outright; otherwise
        # fall back to stderr then stdout, gated by the error-signal requirement.
        candidates: list[str] = []
        if engine_log:
            candidates.append(engine_log)
        else:
            for tail in (stderr_tail, stdout_tail):
                if tail and (not require_error_signal or self._looks_like_error(tail)):
                    candidates.append(tail)

        for candidate in candidates:
            source, excerpt = self._extract_error_source(candidate)
            if source or excerpt:
                if source:
                    self._error_source_file = source
                if excerpt:
                    self._error_excerpt = excerpt
                break

    # Matches file references in engine/interpreter tracebacks, most-specific first:
    #   Python/Ren'Py:  File "game/script.rpy", line 123, in foo
    #   Godot:          res://scene.gd:45  (also at: ... (res://foo.gd:12))
    _FILE_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
    _GODOT_FRAME_RE = re.compile(r'(res://[^\s:"\')]+\.[A-Za-z0-9]+):(\d+)')

    def _extract_error_source(self, text: str | None) -> tuple[str | None, str | None]:
        """Extract the most relevant source file and error excerpt from a traceback.

        Prefers the *last* file frame — the one closest to where the failure
        actually occurred — and keeps the trailing lines as the excerpt.

        Returns:
            ``(source_file, excerpt)``; either element may be ``None``.
        """
        if not text or not text.strip():
            return None, None

        lines = [ln for ln in text.splitlines() if ln.strip()]
        excerpt = "\n".join(lines[-40:]) if lines else None

        source_file: str | None = None
        # Last match wins (closest to the failure). Check both conventions.
        py_matches = self._FILE_FRAME_RE.findall(text)
        godot_matches = self._GODOT_FRAME_RE.findall(text)
        if py_matches:
            source_file = py_matches[-1][0]
        elif godot_matches:
            file_path, line_no = godot_matches[-1]
            source_file = f"{file_path}:{line_no}"

        return source_file, excerpt

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _startup_countdown(self) -> None:
        """Print a 3-second countdown with failsafe reminder."""
        print("\n" + "=" * 60)
        print("⚠️  AUTOMATED INPUT STARTING IN 3 SECONDS")
        print("   Move mouse to a screen corner to abort (failsafe).")
        print("   Press Ctrl+C to interrupt.")
        print("=" * 60)
        for i in range(3, 0, -1):
            print(f"   Starting in {i}…", end="\r", flush=True)
            time.sleep(1)
        print("   ▶ RUNNING.                          ")

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def analyze_and_act(self) -> str:
        """Single capture → analyse → act → log cycle.

        Returns:
            One of ``"completed"``, ``"clicked"``, ``"waited"``, ``"typed"``,
            ``"pressed"``, or ``"unknown"``.
        """
        # 1. Capture
        try:
            pil_image = self.capturer.capture_pil()
            # Overlay a labeled coordinate grid so the model can read click
            # coordinates off reference lines. The grid is drawn at true pixel
            # positions, so returned coordinates need no extra transform.
            if self.settings.harness.coordinate_grid:
                model_image = Capturer.draw_coordinate_grid(
                    pil_image, spacing=self.settings.harness.grid_spacing,
                )
            else:
                model_image = pil_image
            # Encode to base64 for the LLM
            buf = BytesIO()
            model_image.save(buf, format="PNG")
            import base64
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except ScreenCaptureError as exc:
            logger.error("Step %d/%d | 💥 Screen capture failed: %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"

        # 2. Build prompts
        # Build step history for the LLM with rich per-step context.
        # The window must be wide enough to span a full game-loop cycle so the
        # model remembers which screens it has already visited.
        history_entries = self.step_history
        window = self.settings.logging.step_history_window
        visible = history_entries[-window:] if window > 0 and history_entries else []
        context_str = "\n".join(visible) if visible else "(no prior steps — starting playthrough)"

        # Inject a stuck-state warning when the harness has detected repetition.
        # This makes the VLM a participant in recovery rather than a passive
        # recipient of key-press recovery strategies.
        stuck_warning = ""
        if self.stuck_counter >= 2:
            stuck_warning = (
                f"\n\n⚠️  STUCK-STATE WARNING: You have repeated the same action type "
                f"('{self.last_action}') {self.stuck_counter + 1} times in a row. "
                f"These attempts appear to have had no effect. Pick a DIFFERENT action "
                f"or target a DIFFERENT element. Do NOT repeat the last action.\n"
            )
        # Inject a save-reminder hint when approaching the save-trigger step.
        # This gives the LLM one full step's context to navigate to the save
        # screen before the actual save-trigger step fires.
        save_hint = ""
        if self.save_id and not self._save_triggered:
            if self.step == self._save_trigger_step - 1:
                # One step before — hint that a save is coming.
                save_hint = (
                    "\n\n💾 SAVE-PHASE NOTICE: The next step will be the final action "
                    "before this session pauses. Navigate towards a screen where you "
                    "can click a Save / Save Game option. You do not need to save yet — "
                    "just get close to the save UI.\n"
                )
            elif self.step == self._save_trigger_step:
                # This is the save step — explicitly request clicking the save UI.
                save_hint = (
                    "\n\n💾 SAVE-PHASE ACTION: This is the last action before this "
                    "session saves its state and stops. Click the Save / Save Game "
                    "button (or equivalent) now to preserve the in-game progress. "
                    "After this step the harness will serialize its state and cleanly "
                    "stop, resuming from this point on the next run with the same --save-id.\n"
                )
        context_str = context_str + stuck_warning + save_hint

        # Mark the save as triggered *before* the LLM call so the prompt
        # injection only fires once. If the LLM call fails at the save step,
        # we still flag it to avoid re-triggering on a retry/crash recovery.
        if save_hint and self.step == self._save_trigger_step:
            self._save_triggered = True

        system_prompt = make_system_prompt(self.settings.system_prompt)
        user_prompt = make_user_prompt(context_str, self.settings.user_prompt_template)

        # 3. Analyse via LLM (timed), with the current adaptively-throttled budget.
        llm_start = time.time()
        try:
            raw = self.llm.analyze(
                image_b64, system_prompt, user_prompt,
                max_tokens=self._effective_max_tokens,
            )
        except RuntimeError as exc:
            logger.error("Step %d/%d | 💥 LLM call failed: %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"
        duration_ms = int((time.time() - llm_start) * 1000)

        # 3.5 Adaptive reasoning throttling — after a successful response,
        # check the reasoning-to-output ratio and tighten the budget if the
        # model spent most of its allowance on thinking tokens.
        self._apply_reasoning_throttle(raw)

        # 4. Validate with Pydantic
        try:
            response = ActionResponse.model_validate(raw)
        except Exception as exc:
            logger.warning("Step %d/%d | ⚠️  Invalid LLM response (schema): %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"

        # 5. Debug: log full parsed ActionResponse
        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | ActionResponse: action=%s | text=%s | key=%s | narrative=%s | reasoning=%s",
                self.step,
                self.settings.harness.max_steps,
                response.action.value,
                response.text_to_type,
                response.key_to_press,
                response.narrative,
                response.reasoning,
            )

        # 6. Log narrative
        logger.info(
            "📝 Step %d/%d | %s",
            self.step,
            self.settings.harness.max_steps,
            response.narrative,
        )
        # Record a rich step entry: action type + bounding box centre so the
        # model can see exactly what was done and where clicks landed.
        action_label = response.action.value.upper()
        coord_detail = ""
        if response.action.value == "click":
            bb = response.bounding_box
            if bb is not None and len(bb) == 4:
                cx = int((bb[0] + bb[2]) / 2)
                cy = int((bb[1] + bb[3]) / 2)
                coord_detail = f" at bbox-centre ({cx}, {cy})"
        entry = f"Step {self.step} [{action_label}{coord_detail}]: {response.narrative}"
        self.step_history.append(entry)
        self._write_log_entry(response, image_b64, duration_ms)

        # 7. Generate debug overlay image (when debug_screen is enabled).
        # Draw onto the gridded image the model actually saw, so the debug
        # PNG shows the same reference lines plus where the click landed.
        if self.debug_screen:
            self._save_debug_overlay(model_image, response)

        # 8. Execute
        return self._execute(response)

    # ------------------------------------------------------------------
    # Adaptive reasoning throttling
    # ------------------------------------------------------------------

    def _apply_reasoning_throttle(self, raw: dict) -> None:
        """Tighten the per-call token budget when the model burns budget on thinking.

        Reads ``_reasoning_chars`` from the raw LLM response dict (set by
        ``OpenAIClient._request``). When the reasoning-trace length exceeds
        the ``min_reasoning_ratio`` threshold of the current budget, the
        harness steps down to the next lower budget tier for subsequent calls.

        This creates a negative feedback loop: overthinking → smaller budget →
        less room to think → faster decisions.
        """

        # Approximate token count from character count (rough: 4 chars ≈ 1 token).
        reasoning_chars = raw.get("_reasoning_chars", 0)
        if reasoning_chars <= 0:
            return

        # Rough token estimate: ~4 chars per token.
        reasoning_tokens_est = max(1, reasoning_chars // 4)
        budget = max(self._effective_max_tokens, 1)
        ratio = reasoning_tokens_est / budget

        threshold = self.settings.llm.min_reasoning_ratio
        if ratio < threshold:
            return

        # Step down through the budget tiers.
        tiers = [600, 450, 300, 200]
        current = self._effective_max_tokens
        # Find the next lower tier (skip tiers that are >= current).
        next_budget = None
        for t in tiers:
            if t < current:
                next_budget = t
                break

        if next_budget is None:
            # Already at the floor — nothing more we can do.
            return

        self._effective_max_tokens = next_budget
        self._reasoning_penalties += 1
        logger.warning(
            "🧠 Step %d/%d | Reasoning throttling: ~%d reasoning tokens "
            "exceed %.0f%% of the %d-token budget — tightening to %d tokens "
            "(penalty #%d).",
            self.step,
            self.settings.harness.max_steps,
            reasoning_tokens_est,
            threshold * 100,
            current,
            next_budget,
            self._reasoning_penalties,
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, response: ActionResponse) -> str:
        action = response.action.value

        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | Executing action='%s' | dry_run=%s",
                self.step, self.settings.harness.max_steps, action, self.dry_run,
            )

        # Stuck detection
        if action == self.last_action:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_action = action

        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | stuck_counter=%d (threshold=%d) | last_action='%s'",
                self.step, self.settings.harness.max_steps,
                self.stuck_counter, self.settings.harness.stuck_threshold,
                self.last_action,
            )

        if self.stuck_counter >= self.settings.harness.stuck_threshold:
            logger.warning(
                "Step %d/%d | ⚠️  Possible stuck state — %d identical '%s' actions.",
                self.step,
                self.settings.harness.max_steps,
                self.stuck_counter,
                action,
            )

            # Attempt automatic recovery if enabled
            if self.settings.harness.stuck_recovery:
                recovery_result = self._recover_from_stuck()
                if recovery_result is not None:
                    return recovery_result

        # Track action counts for summary
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        if action == "done":
            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'done' → returning 'completed'",
                    self.step, self.settings.harness.max_steps,
                )
            logger.info("Step %d/%d | 🏁 Game ended or returned to main menu.", self.step, self.settings.harness.max_steps)
            return "completed"

        if action == "wait":
            wait_time = self.settings.harness.wait_duration
            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'wait' → polling up to %.1fs for stability",
                    self.step, self.settings.harness.max_steps, wait_time,
                )
            logger.info("Step %d/%d | ⏳ Waiting up to %ss for scene to stabilise…", self.step, self.settings.harness.max_steps, wait_time)
            if not self.dry_run:
                self._wait_until_stable(wait_time)
            return "waited"

        if action == "advance":
            return self._advance_macro(response)

        if action == "click":
            # Resolve click target: bounding_box centre
            bb = response.bounding_box
            if bb is not None and len(bb) == 4:
                # Centre of bounding box: [x1, y1, x2, y2]
                raw_x = (bb[0] + bb[2]) / 2.0
                raw_y = (bb[1] + bb[3]) / 2.0
                target_source = f"b-box centre from [{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]"
            else:
                logger.warning("Step %d/%d | ⚠️  'click' action without bounding_box — skipping.", self.step, self.settings.harness.max_steps)
                return "unknown"

            # Capture a cheap signature of the screen *before* the click so we
            # can verify the click actually did something, and fall back to the
            # model's other ranked candidates if it didn't — all without a new
            # VLM call. Returns None in dry-run or when capture is unavailable.
            pre_sig = self._capture_signature()

            if not self._perform_click(raw_x, raw_y, target_source):
                return "unknown"

            self._verify_click_or_fallback(response, pre_sig, (raw_x, raw_y))
            return "clicked"

        if action == "type":
            text = response.text_to_type or ""
            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'type' | text='%s' | length=%d",
                    self.step, self.settings.harness.max_steps, text, len(text),
                )
            if not self.dry_run:
                pyautogui.write(text, interval=0.05)
                logger.info("Step %d/%d | ⌨️  Typed: %s", self.step, self.settings.harness.max_steps, text)
            else:
                logger.info("Step %d/%d | 🔍 [DRY RUN] Would type: %s", self.step, self.settings.harness.max_steps, text)
            return "typed"

        if action == "press":
            key = response.key_to_press or ""
            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'press' | key='%s'",
                    self.step, self.settings.harness.max_steps, key,
                )
            if not key:
                logger.warning("Step %d/%d | ⚠️  'press' action without key_to_press — skipping.", self.step, self.settings.harness.max_steps)
                return "unknown"
            if not self.dry_run:
                pyautogui.press(key)
                logger.info("Step %d/%d | ⌨️  Pressed key: %s", self.step, self.settings.harness.max_steps, key)
            else:
                logger.info("Step %d/%d | 🔍 [DRY RUN] Would press key: %s", self.step, self.settings.harness.max_steps, key)
            return "pressed"

        logger.warning("Step %d/%d | ⚠️  Unknown action: %s", self.step, self.settings.harness.max_steps, action)
        return "unknown"

    # ------------------------------------------------------------------
    # Frame-change detection helpers (VLM-call reduction, verify-gated)
    # ------------------------------------------------------------------

    def _capture_signature(self):
        """Capture the current frame and return its cheap change-signature.

        Returns ``None`` in dry-run, or whenever a real ``PIL.Image`` cannot be
        produced (e.g. a mocked capturer in tests) so callers transparently
        skip verification rather than acting on a bogus comparison.
        """
        if self.dry_run:
            return None
        try:
            img = self.capturer.capture_pil()
            sig = frame_signature(img)
        except Exception:
            return None
        return sig if isinstance(sig, Image.Image) else None

    def _frame_changed_since(self, pre_sig) -> bool:
        """Poll a few times for the screen to differ from *pre_sig*.

        Gives the game a moment to respond after an input. Returns ``True`` as
        soon as a captured frame exceeds ``frame_change_min_delta`` from
        *pre_sig*; ``False`` if it stays unchanged across
        ``advance_stable_polls`` (+1) polls. When a frame can't be captured we
        conservatively report ``True`` (assume it changed) so we don't fire
        spurious fallback clicks.
        """
        h = self.settings.harness
        interval = h.advance_poll_interval
        for _ in range(h.advance_stable_polls + 1):
            if interval > 0:
                time.sleep(interval)
            sig = self._capture_signature()
            if sig is None:
                return True
            if frame_delta(pre_sig, sig) >= h.frame_change_min_delta:
                return True
        return False

    def _verify_click_or_fallback(self, response: ActionResponse, pre_sig, attempted) -> None:
        """Confirm a click had an effect; else try the next ranked candidate.

        The model already enumerates up to three ranked ``candidates`` per
        frame. When the chosen click leaves the screen unchanged, we walk that
        existing list and try the next click-type candidate — reusing work
        we've already paid for instead of spending a fresh VLM call. Only when
        every candidate is exhausted does control return to the caller (which
        proceeds to stuck recovery / a new VLM parse on the next step).
        """
        h = self.settings.harness
        if pre_sig is None or not h.candidate_fallback:
            return
        if self._frame_changed_since(pre_sig):
            return  # the click worked — nothing to do

        attempted_pts = [attempted]
        for cand in response.candidates:
            if cand.action != ActionType.CLICK:
                continue
            cb = cand.bbox
            if cb is None or len(cb) != 4:
                continue
            cx = (cb[0] + cb[2]) / 2.0
            cy = (cb[1] + cb[3]) / 2.0
            # Skip candidates whose centre coincides with one we've already tried.
            if any(abs(cx - px) < 5 and abs(cy - py) < 5 for px, py in attempted_pts):
                continue
            logger.info(
                "Step %d/%d | 🎯 Click had no visible effect — trying next candidate '%s' (no new VLM call).",
                self.step, self.settings.harness.max_steps, cand.label,
            )
            if not self._perform_click(cx, cy, f"candidate '{cand.label}'"):
                continue
            attempted_pts.append((cx, cy))
            if self._frame_changed_since(pre_sig):
                logger.info(
                    "Step %d/%d | ✅ Recovered via candidate '%s'.",
                    self.step, self.settings.harness.max_steps, cand.label,
                )
                return
        logger.info(
            "Step %d/%d | ⚠️  No ranked candidate changed the screen — deferring to recovery / next VLM parse.",
            self.step, self.settings.harness.max_steps,
        )

    def _perform_click(self, raw_x: float, raw_y: float, target_source: str) -> bool:
        """Scale, bounds-check and execute a single click at model-space coords.

        Returns ``True`` if the click was issued (or would be, in dry-run),
        ``False`` if the coordinates fell outside the screen and were skipped.
        Shared by the primary click path and candidate fallback.
        """
        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | Click | target=%s | raw=(%.1f, %.1f) | capturer.scale=%.4f",
                self.step, self.settings.harness.max_steps,
                target_source, raw_x, raw_y, self.capturer.scale,
            )

        x, y = self.capturer.scale_coordinates(raw_x, raw_y)

        if not self._validate_coordinates(x, y):
            screen_w, screen_h = pyautogui.size()
            logger.warning(
                "Step %d/%d | ⚠️  Click coordinates (%d, %d) are outside expected screen bounds — skipping.",
                self.step, self.settings.harness.max_steps, x, y,
            )
            return False

        if not self.dry_run:
            self.capturer.click(x, y)
            logger.info("Step %d/%d | 🖱️  Clicked (%d, %d) [scale=%.2f, %s]", self.step, self.settings.harness.max_steps, x, y, self.capturer.scale, target_source)
        else:
            logger.info("Step %d/%d | 🔍 [DRY RUN] Would click (%d, %d) [%s]", self.step, self.settings.harness.max_steps, x, y, target_source)
        # Feed the actual click coordinates back to the LLM context so it can
        # self-correct if clicks are landing in the wrong place.
        self.step_history.append(
            f"  ↳ Click executed at absolute screen coordinates ({x}, {y}) [{target_source}]."
        )
        return True

    def _wait_until_stable(self, ceiling: float) -> None:
        """Poll until the screen settles, capped at *ceiling* seconds.

        Replaces a fixed sleep: returns early once the frame stops changing
        (``advance_stable_polls`` consecutive sub-threshold polls) so fast
        scenes don't over-wait, while slow ones aren't parsed mid-animation.
        Falls back to sleeping the remaining time if frames can't be captured.
        """
        h = self.settings.harness
        interval = h.advance_poll_interval if h.advance_poll_interval > 0 else 0.3
        deadline = time.time() + ceiling
        prev_sig = self._capture_signature()
        stable = 0
        while time.time() < deadline:
            time.sleep(interval)
            sig = self._capture_signature()
            if prev_sig is None or sig is None:
                remaining = deadline - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                return
            if frame_delta(prev_sig, sig) < h.frame_change_min_delta:
                stable += 1
                if stable >= h.advance_stable_polls:
                    return
            else:
                stable = 0
            prev_sig = sig

    def _advance_macro(self, response: ActionResponse) -> str:
        """Fast-forward plain dialogue with per-repeat frame verification.

        One ``advance`` decision from the model collapses into several advance
        inputs here, so the harness doesn't pay a full VLM round-trip per line.
        Each repeat is verify-gated: we keep going only while the frame keeps
        changing in a text-only ('minor') way, and stop the moment it changes
        structurally (a choice/scene → hand back to the VLM) or stops changing.
        Intermediate frames are still saved for visual-defect coverage.
        """
        h = self.settings.harness
        key = h.advance_key

        if self.dry_run:
            logger.info(
                "Step %d/%d | 🔍 [DRY RUN] Would advance dialogue (press '%s', up to %d×).",
                self.step, self.settings.harness.max_steps, key, h.advance_max_repeats,
            )
            return "advanced"

        self._macro_frame_seq = 0
        prev_sig = self._capture_signature()
        progressed = 0
        stable = 0
        for i in range(h.advance_max_repeats):
            pyautogui.press(key)
            if h.advance_poll_interval > 0:
                time.sleep(h.advance_poll_interval)
            cur_img, cur_sig = self._capture_frame_and_sig()
            if h.save_macro_frames and cur_img is not None:
                self._save_macro_frame(cur_img)

            if prev_sig is None or cur_sig is None:
                # Can't verify — issue a single advance and hand back to the VLM.
                break

            change = classify_change(
                frame_delta(prev_sig, cur_sig),
                h.frame_change_min_delta,
                h.frame_change_structural_delta,
            )
            if change == "structural":
                logger.info(
                    "Step %d/%d | ⏩ Advance: structural change after %d step(s) — new decision point, handing back to VLM.",
                    self.step, self.settings.harness.max_steps, i + 1,
                )
                break
            if change == "none":
                stable += 1
                if stable >= h.advance_stable_polls:
                    logger.info(
                        "Step %d/%d | ⏩ Advance: screen settled after %d step(s) — dialogue ended, handing back to VLM.",
                        self.step, self.settings.harness.max_steps, i + 1,
                    )
                    break
            else:  # "minor" — dialogue still rolling
                stable = 0
                progressed += 1
            prev_sig = cur_sig

        logger.info(
            "Step %d/%d | ⏩ Advance macro fast-forwarded %d dialogue line(s).",
            self.step, self.settings.harness.max_steps, progressed,
        )
        # Advancing dialogue is legitimate repetition — don't let it trip the
        # stuck detector when it actually made progress.
        if progressed > 0:
            self.stuck_counter = 0
        return "advanced"

    def _capture_frame_and_sig(self):
        """Capture the current frame, returning ``(image, signature)``.

        Both are ``None`` in dry-run or when a real image can't be captured.
        Used by the advance macro so a single capture serves both the change
        check and the saved-frame fidelity record.
        """
        if self.dry_run:
            return None, None
        try:
            img = self.capturer.capture_pil()
            sig = frame_signature(img)
        except Exception:
            return None, None
        if not isinstance(sig, Image.Image):
            return None, None
        return img, sig

    def _save_macro_frame(self, img: Image.Image) -> None:
        """Persist an intermediate advance-macro frame for defect coverage."""
        try:
            screenshot_dir = self.settings.logging.screenshot_dir
            os.makedirs(screenshot_dir, exist_ok=True)
            run_prefix = self.run_id[:5]
            filename = f"{run_prefix}_step_{self.step:04d}_adv_{self._macro_frame_seq:02d}.png"
            img.save(os.path.join(screenshot_dir, filename))
            self._macro_frame_seq += 1
        except Exception as exc:
            logger.warning(
                "Step %d/%d | ⚠️  Failed to save advance-macro frame: %s",
                self.step, self.settings.harness.max_steps, exc,
            )

    def _validate_coordinates(self, x: int, y: int) -> bool:
        """Check whether coordinates fall within the expected screen region.

        When a game window is bound, ``x`` and ``y`` are absolute logical
        screen coordinates (window-relative + window offset).  Validation
        checks against the logical screen size rather than the game
        resolution, because the coordinates are already transformed to the
        screen coordinate space by ``scale_coordinates()``.
        """
        import pyautogui

        screen_w, screen_h = pyautogui.size()
        # Allow a small margin for edge clicks
        margin = 50
        return (-margin <= x <= screen_w + margin) and (-margin <= y <= screen_h + margin)

    # ------------------------------------------------------------------
    # Save / resume state
    # ------------------------------------------------------------------

    @property
    def _save_trigger_step(self) -> int:
        """The step number at which the harness should inject the save prompt.

        Computed as ``max_steps - save_before_end_steps``, clamped to >= 1.
        """
        offset = self.settings.harness.save_before_end_steps
        return max(1, self.settings.harness.max_steps - offset)

    def _save_state_path(self) -> Path:
        """Return the path to the save-state file for this session."""
        return self.runs_dir / self.save_id / "save_state.json"  # type: ignore[operator]

    def _load_save_state(self) -> bool:
        """Load a previous session's state from disk and restore harness internals.

        Returns:
            ``True`` if a save state was loaded, ``False`` if no save file exists
            or it could not be parsed.
        """
        path = self._save_state_path()
        if not path.is_file():
            logger.info(
                "💾 Save state '%s' not found — starting fresh session.", self.save_id,
            )
            return False

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            state = SaveState.model_validate(raw)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("💾 Could not parse save state '%s': %s — starting fresh.", self.save_id, exc)
            return False

        # Restore harness internals from the serialised state.
        self.step = state.step
        self.step_history = list(state.step_history)
        self.stuck_counter = state.stuck_counter
        self.last_action = state.last_action
        self.action_counts = dict(state.action_counts)
        self._total_steps_across_runs = state.total_steps_across_runs
        # The step variable itself holds the *start* of this run's range;
        # the main loop iterates from step+1 onwards.
        logger.info(
            "💾 Resumed session '%s' | prior runs: %d | cumulative steps: %d | "
            "starting at step %d (this run's step counter)",
            self.save_id,
            len(state.runs),
            state.total_steps_across_runs,
            self.step,
        )

        # Append a boundary marker so the LLM knows consecutive runs are a
        # continuation, not a glitch.
        self.step_history.append(
            f"--- End of previous run ({state.last_run_id or 'unknown'}). "
            f"Continuing in a new run session. The game screen should be at the "
            f"same position. ---"
        )
        return True

    def _write_save_state(self) -> None:
        """Serialize the current harness state to the save-state file.

        Called at the end of a run when ``--save-id`` is set. The saved step
        is the last completed step so the next run picks up at ``step + 1``.
        """
        path = self._save_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state to accumulate run history and carry forward the
        # cumulative step count (if a prior run wrote one).
        runs: list[str] = []
        prior_total = self._total_steps_across_runs
        if path.is_file():
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
                runs = prev.get("runs", [])
                # Carry forward the prior total so each run adds on top.
                prior_total = max(prior_total, prev.get("total_steps_across_runs", 0))
            except (json.JSONDecodeError, Exception):
                pass

        if self.run_id not in runs:
            runs.append(self.run_id)

        # Cumulative step count: prior total + steps completed this run.
        total = prior_total + self.step

        state = SaveState(
            save_id=self.save_id,  # type: ignore[arg-type]
            last_run_id=self.run_id,
            runs=runs,
            step=self.step,
            step_history=self.step_history,
            stuck_counter=self.stuck_counter,
            last_action=self.last_action,
            action_counts=self.action_counts,
            total_steps_across_runs=total,
            game_save_triggered=self._save_triggered,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        logger.info(
            "💾 Wrote save state '%s' at step %d | total across runs: %d | runs: %s",
            self.save_id, self.step, total, ", ".join(runs),
        )

    # ------------------------------------------------------------------
    # Stuck-state recovery
    # ------------------------------------------------------------------

    def _recover_from_stuck(self) -> str | None:
        """Attempt to break out of a detected stuck state.

        Rotates through a set of recovery strategies. Each call tries the
        next strategy so the harness doesn't spam the same recovery action.

        Returns the action string if a recovery action was executed,
        ``None`` if recovery is disabled or no strategies remain.
        """
        # Build the strategy list once
        if not self._recovery_strategies:
            self._recovery_strategies = [
                "press:escape",
                "press:space",
                "click_center",
                "wait:5.0",
                "press:enter",
            ]

        strategy_idx = self._recovery_attempts % len(self._recovery_strategies)
        strategy = self._recovery_strategies[strategy_idx]
        self._recovery_attempts += 1

        logger.info(
            "🔄 Step %d/%d | Stuck recovery attempt #%d — strategy: '%s'",
            self.step, self.settings.harness.max_steps,
            self._recovery_attempts, strategy,
        )

        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | Recovery: strategy_idx=%d, recovery_attempts=%d",
                self.step, self.settings.harness.max_steps,
                strategy_idx, self._recovery_attempts,
            )

        if self.dry_run:
            logger.info(
                "Step %d/%d | 🔍 [DRY RUN] Would execute recovery: %s",
                self.step, self.settings.harness.max_steps, strategy,
            )
            # In dry-run, pretend it worked and reset stuck state
            self.stuck_counter = 0
            self._recovery_attempts = 0
            self._recovery_strategies = []
            return "recovery"

        if strategy.startswith("press:"):
            key = strategy.split(":", 1)[1]
            pyautogui.press(key)
            logger.info(
                "Step %d/%d | 🔄 Recovery: pressed '%s' key",
                self.step, self.settings.harness.max_steps, key,
            )
            # Track as a recovery action
            self.action_counts["recovery"] = self.action_counts.get("recovery", 0) + 1
            # Reset stuck counter since we did a recovery action
            self.stuck_counter = 0
            return "recovery"

        elif strategy == "click_center":
            w, h = self.settings.game.resolution
            cx, cy = w // 2, h // 2
            self.capturer.click(cx, cy)
            logger.info(
                "Step %d/%d | 🔄 Recovery: clicked center (%d, %d)",
                self.step, self.settings.harness.max_steps, cx, cy,
            )
            self.action_counts["recovery"] = self.action_counts.get("recovery", 0) + 1
            self.stuck_counter = 0
            return "recovery"

        elif strategy.startswith("wait:"):
            wait_secs = float(strategy.split(":", 1)[1])
            logger.info(
                "Step %d/%d | 🔄 Recovery: extended wait %.1fs",
                self.step, self.settings.harness.max_steps, wait_secs,
            )
            time.sleep(wait_secs)
            self.action_counts["recovery"] = self.action_counts.get("recovery", 0) + 1
            self.stuck_counter = 0
            return "recovery"

        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log_entry(
        self, response: ActionResponse, image_b64: str, duration_ms: int = 0
    ) -> None:
        """Append a structured log entry to the JSONL file and optionally save a screenshot."""
        log_settings = self.settings.logging
        screenshot_path: str | None = None

        if log_settings.save_screenshots:
            run_prefix = self.run_id[:5]
            filename = f"{run_prefix}_step_{self.step:04d}.png"
            screenshot_path = os.path.join(log_settings.screenshot_dir, filename)
            try:
                img = self.capturer.capture_pil()
                img.save(screenshot_path)
            except Exception as exc:
                logger.warning("Step %d/%d | ⚠️  Failed to save screenshot: %s", self.step, self.settings.harness.max_steps, exc)
                screenshot_path = None

        entry = LogEntry(
            step=self.step,
            timestamp=datetime.now().isoformat(),
            llm_response=response,
            duration_ms=duration_ms,
            image_hex_truncated=image_b64[:200] + "...",
            screenshot_path=screenshot_path,
        )

        with open(log_settings.log_file, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def _save_debug_overlay(self, image: Image.Image, response: ActionResponse) -> None:
        """Draw debug annotations on *image* and save to the run directory.

        Only called when ``debug_screen`` is enabled and ``run_id`` is set.
        Produces ``debug_step_NNNN.png`` files in the run directory.
        """
        try:
            # Determine bounding box and click point from the response.
            # Model-space coordinates must be mapped to capture-pixel space
            # via _to_pixel_space (calibration / coordinate_max normalisation)
            # so the overlay crosshair and bbox land on the correct image
            # pixels — the same transform that scale_coordinates applies
            # before the DPI scale and window offset for the real click.
            bounding_box: list[float] | None = None
            click_point: tuple[int, int] | None = None

            bb = response.bounding_box
            if bb is not None and len(bb) == 4:
                # Transform model-space corners to capture-pixel space.
                px1, py1 = self.capturer._to_pixel_space(bb[0], bb[1])
                px2, py2 = self.capturer._to_pixel_space(bb[2], bb[3])
                bounding_box = [px1, py1, px2, py2]
                cx = int((px1 + px2) / 2)
                cy = int((py1 + py2) / 2)
                click_point = (cx, cy)

            # Build metadata dict
            metadata: dict[str, str] = {
                "Step": str(self.step),
                "Action": response.action.value,
            }
            if response.bounding_box:
                metadata["BBox"] = (
                    f"[{response.bounding_box[0]:.0f}, {response.bounding_box[1]:.0f}, "
                    f"{response.bounding_box[2]:.0f}, {response.bounding_box[3]:.0f}]"
                )
            if click_point:
                metadata["ClickPt (img)"] = str(click_point)
            metadata["Scale"] = f"{self.capturer.scale:.4f}"
            if self.capturer.game_window:
                gw = self.capturer.game_window
                metadata["Window"] = f"{gw.x},{gw.y} {gw.width}x{gw.height}"

            # Draw the overlay
            annotated = Capturer.draw_debug_overlay(
                image,
                bounding_box=bounding_box,
                click_point=click_point,
                metadata=metadata,
            )

            # Save to the namespaced run directory.
            out_dir = self.runs_dir / self.run_id  # type: ignore[arg-type]
            os.makedirs(out_dir, exist_ok=True)
            run_prefix = self.run_id[:5]
            out_path = out_dir / f"{run_prefix}_debug_step_{self.step:04d}.png"
            annotated.save(str(out_path), format="PNG")
            logger.info("📸 Debug overlay saved → %s", out_path)

        except Exception as exc:
            logger.warning("⚠️  Failed to save debug overlay for step %d: %s", self.step, exc)

    def _log_summary(self, report: dict | None = None) -> None:
        """Log a run summary at the end of the playthrough.

        Args:
            report: The next-action report dict, rendered into the summary and
                echoed to stdout so it is the single salient takeaway.
        """
        duration = time.time() - self.start_time if self.start_time else 0
        actions_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.action_counts.items())
        ) or "none"
        logger.info("=" * 60)
        logger.info("📊 RUN SUMMARY")
        logger.info("=" * 60)
        logger.info("  Steps completed : %d", self.step)
        logger.info("  Completion      : %s", self.completion_reason or "unknown")
        logger.info("  Duration        : %.1fs", duration)
        logger.info("  Actions         : %s", actions_str)
        logger.info("  Log file        : %s", self.settings.logging.log_file)
        if report:
            related = report.get("related_steps") or []
            related_str = f" (step {', '.join(map(str, related))})" if related else ""
            logger.info("  Next action     : %s%s", report.get("next_action", ""), related_str)
        logger.info("=" * 60)

        if report:
            self._print_next_action(report)

    def _print_next_action(self, report: dict) -> None:
        """Echo the single salient takeaway to stdout as the final block."""
        related = report.get("related_steps") or []
        related_str = f"  (step {', '.join(map(str, related))})" if related else ""
        print("\n" + "=" * 60)
        overall = (report.get("overall_verdict") or "").strip()
        if overall:
            print("🎭 PLAYER EXPERIENCE:")
            print(f"   {overall}")
            print("-" * 60)
        print(f"➡️  NEXT ACTION:{related_str}")
        print(f"   {(report.get('next_action') or '').strip()}")
        src = report.get("error_source_file")
        if src:
            print(f"   File: {src}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, next_action: dict | None = None) -> None:
        """Write a ``run_manifest.json`` describing this run.

        The manifest is the canonical entry point for the ingestion system:
        discover runs by globbing ``runs/*/run_manifest.json``.  Every run
        gets a manifest because ``run_id`` is always set (explicit or
        auto-generated).

        Args:
            next_action: The next-action report dict to embed. Mirrored into the
                manifest so the ingestion system reads the same takeaway shown in
                ``NEXT_ACTION.md`` and on stdout.
        """
        from . import __version__  # local import to avoid circular at module load

        duration = time.time() - self.start_time if self.start_time else 0
        g = self.settings.game
        manifest = {
            "run_id": self.run_id,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat()
            if self.start_time
            else None,
            "ended_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "exit_code": self._exit_code,
            "completion_reason": self.completion_reason,
            "game": {
                "type": g.type,
                "path": g.path,
                "resolution": list(g.resolution),
            },
            "git": self._git_info or {"commit": None, "branch": None, "dirty": None},
            "llm": {
                "model": self.settings.llm.model,
                "api_base": self.settings.llm.api_base,
            },
            "steps_completed": self.step,
            "action_counts": dict(self.action_counts),
            "log_file": self.settings.logging.log_file,
            "screenshot_dir": self.settings.logging.screenshot_dir,
            "tester_version": __version__,
        }

        if next_action:
            manifest["next_action"] = next_action
            # Back-compat: older readers expect a recap with key_complaint.
            manifest["recap"] = {
                "key_complaint": next_action.get("summary")
                or next_action.get("next_action", "")
            }

        run_dir = self.runs_dir / self.run_id
        os.makedirs(run_dir, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            logger.info("📦 Wrote run manifest: %s", manifest_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"⚠️  Failed to write run manifest: {exc}")

    # ------------------------------------------------------------------
    # Next-action takeaway
    # ------------------------------------------------------------------

    def _build_next_action(self) -> dict:
        """Build the single 'next key action' takeaway for this run.

        Hybrid strategy:
          * Hard errors (game death, harness crash, config/runtime error) are
            summarised deterministically from the concrete error text captured
            during the run — no LLM call, so the case that matters most never
            depends on the LLM succeeding.
          * Soft/gameplay outcomes (completed, max-steps, stuck, interrupted)
            are summarised by an LLM recap over the playthrough log.

        Always returns a serialisable dict (never ``None``).
        """
        # Hard-error branch first, so a zero-step config error still produces a
        # meaningful takeaway before the "no steps" short-circuit below.
        hard = self._hard_error_action()
        if hard is not None:
            logger.info("📋 Next action | %s", hard["next_action"])
            return hard

        report = self._soft_action()
        logger.info("📋 Next action | %s", report["next_action"])
        return report

    def _hard_error_action(self) -> dict | None:
        """Deterministic takeaway for hard errors, or ``None`` if not one."""
        reason = self.completion_reason or ""
        is_game_died = reason == "game_died"
        is_crash = reason.startswith("crashed:")
        is_config = self._exit_code == EXIT_CONFIG_ERROR
        is_runtime = self._exit_code == EXIT_RUNTIME_ERROR

        if not (is_game_died or is_crash or is_config or is_runtime):
            return None

        error_text = self._error_excerpt or self._crash_traceback or self._game_stderr_tail
        source = self._error_source_file
        related = [self.step] if self.step else []
        step_phrase = f" at step {self.step}" if self.step else ""

        # Headline = the exception/message line (last non-empty line of a
        # traceback), which is the most informative single line.
        headline = ""
        if error_text:
            non_empty = [ln.strip() for ln in error_text.splitlines() if ln.strip()]
            if non_empty:
                headline = non_empty[-1]

        if is_game_died:
            status = NextActionStatus.GAME_DIED
            how = f"The game process died unexpectedly{step_phrase}."
        elif is_crash:
            status = NextActionStatus.CRASHED
            how = f"The harness crashed{step_phrase}."
        elif is_config:
            status = NextActionStatus.CONFIG_ERROR
            how = "The run failed during configuration/startup before testing could begin."
        else:
            status = NextActionStatus.CRASHED
            how = f"The run ended with a runtime error{step_phrase}."

        if source:
            next_action = f"Investigate {source}{step_phrase}"
            next_action += f" — {headline}" if headline else f" — {how}"
        elif headline:
            next_action = f"Investigate the failure{step_phrase}: {headline}"
        else:
            next_action = f"Investigate the failure{step_phrase} ({reason or self._exit_code})."

        summary_parts = [how]
        if source:
            summary_parts.append(f"Error references {source}.")
        if headline:
            summary_parts.append(f"Error: {headline}")
        summary = " ".join(summary_parts)

        return NextActionReport(
            next_action=next_action,
            summary=summary,
            error_text=error_text,
            error_source_file=source,
            related_steps=related,
            status=status,
            severity="error",
        ).model_dump(mode="json")

    def _soft_action(self) -> dict:
        """LLM-driven takeaway for soft/gameplay outcomes, with a fallback."""
        status, severity = self._classify_soft_outcome()

        recap = self._llm_recap()
        if recap is not None:
            # Status/severity are decided deterministically here, not by the LLM.
            recap["status"] = status.value
            recap["severity"] = severity
            # Surface any step-level error text the LLM wasn't given/didn't cite.
            if self._error_source_file and not recap.get("error_source_file"):
                recap["error_source_file"] = self._error_source_file
            if self._error_excerpt and not recap.get("error_text"):
                recap["error_text"] = self._error_excerpt
            return recap

        return self._fallback_action(status, severity)

    def _classify_soft_outcome(self) -> tuple[NextActionStatus, str]:
        """Map a soft completion reason to a (status, severity) pair."""
        reason = self.completion_reason or ""
        stuck = self.stuck_counter >= self.settings.harness.stuck_threshold
        if reason == "completed":
            return NextActionStatus.COMPLETED, "info"
        if reason == "interrupted":
            return NextActionStatus.INTERRUPTED, "info"
        if reason == "max_steps_reached":
            return NextActionStatus.MAX_STEPS, "warning"
        if stuck:
            return NextActionStatus.STUCK, "warning"
        return NextActionStatus.UNKNOWN, "info"

    def _fallback_action(self, status: NextActionStatus, severity: str) -> dict:
        """Minimal deterministic takeaway when the LLM recap is unavailable."""
        reason = self.completion_reason or "unknown"
        related = [self.step] if self.step else []
        if status == NextActionStatus.COMPLETED:
            next_action = "No action needed — the playthrough completed naturally."
        elif status == NextActionStatus.MAX_STEPS:
            next_action = (
                f"Review the playthrough — it ran to the step limit ({self.step}) "
                "without reaching a natural end."
            )
        elif status == NextActionStatus.STUCK:
            next_action = (
                f"Review around step {self.step} — the run appears stuck repeating "
                "the same action."
            )
        elif status == NextActionStatus.INTERRUPTED:
            next_action = "Re-run — the playthrough was interrupted before finishing."
        else:
            next_action = f"Review the playthrough log — run ended: {reason}."
        return NextActionReport(
            next_action=next_action,
            summary=f"Run ended ({reason}) after {self.step} step(s).",
            error_text=self._error_excerpt,
            error_source_file=self._error_source_file,
            related_steps=related,
            status=status,
            severity=severity,
        ).model_dump(mode="json")

    def _llm_recap(self) -> dict | None:
        """Send the playthrough log to the LLM for a structured recap.

        Returns:
            A validated :class:`NextActionReport` dict, or ``None`` if the recap
            could not be generated (missing/empty log, LLM error, invalid JSON).
        """
        log_file = self.settings.logging.log_file
        if not os.path.isfile(log_file):
            logger.warning("📋 Recap skipped — no log file found at %s", log_file)
            return None

        if self.step == 0:
            logger.info("📋 Recap skipped — no steps were executed.")
            return None

        # Read all log entries
        try:
            with open(log_file, "r", encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except Exception as exc:
            logger.warning("📋 Recap skipped — could not read log file: %s", exc)
            return None

        if not raw_lines:
            logger.info("📋 Recap skipped — log file is empty.")
            return None

        # Parse and format step log for the recap prompt
        step_lines: list[str] = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            step_num = entry.get("step", "?")
            llm_resp = entry.get("llm_response", {})
            narrative = llm_resp.get("narrative", "")
            action = llm_resp.get("action", "")
            reasoning = llm_resp.get("reasoning", "")
            visual_notes = (llm_resp.get("visual_notes") or "").strip()
            duration = entry.get("duration_ms", 0)
            line = (
                f"Step {step_num} [{action}] ({duration}ms): {narrative}\n"
                f"  Reasoning: {reasoning}"
            )
            if visual_notes:
                line += f"\n  Visual: {visual_notes}"
            step_lines.append(line)

        step_log = "\n".join(step_lines)

        # Build recap prompts
        duration = time.time() - self.start_time if self.start_time else 0
        actions_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.action_counts.items())
        ) or "none"
        personas = self.settings.review.personas
        system_prompt = make_recap_system_prompt()
        user_prompt = make_recap_user_prompt(
            steps_completed=self.step,
            completion_reason=self.completion_reason or "unknown",
            duration=duration,
            action_counts=actions_str,
            step_log=step_log,
            error_context=self._error_excerpt or "",
            personas=personas,
        )

        # The persona-based review is far longer than a per-step action, so give the
        # recap its own budget that scales with how many personas were configured
        # (min one 'Typical player' review). Clamp to the API ceiling.
        n_reviews = max(1, len(personas))
        recap_max_tokens = min(6000, max(1500, 800 + 700 * n_reviews))

        # Call LLM — recap is text-only, so send a minimal 1×1 PNG placeholder
        # to satisfy the vision API contract while keeping focus on the text.
        PLACEHOLDER_PNG = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        try:
            raw = self.llm.summarize(
                PLACEHOLDER_PNG, system_prompt, user_prompt, max_tokens=recap_max_tokens
            )
        except RuntimeError as exc:
            logger.warning("📋 Recap LLM call failed: %s", exc)
            return None

        try:
            report = NextActionReport.model_validate(raw)
        except Exception as exc:
            logger.warning("📋 Recap response validation failed: %s", exc)
            return None

        return report.model_dump(mode="json")

    def _write_next_action(self, report: dict) -> None:
        """Write ``NEXT_ACTION.md`` — the primary human-facing takeaway file.

        Rendered from the same report dict embedded in the manifest and printed
        to stdout, so all three locations agree. Failure is non-fatal.
        """
        lines = [f"# Next Action\n", f"{report.get('next_action', '').strip()}\n"]

        summary = (report.get("summary") or "").strip()
        if summary:
            lines.append("## Summary\n")
            lines.append(f"{summary}\n")

        persona_reviews = report.get("persona_reviews") or []
        overall = (report.get("overall_verdict") or "").strip()
        if persona_reviews or overall:
            lines.append("## Player Experience Review\n")
            for pr in persona_reviews:
                persona = (pr.get("persona") or "Player").strip()
                sentiment = (pr.get("sentiment") or "").strip()
                heading = f"### {persona}"
                if sentiment:
                    heading += f" — _{sentiment}_"
                lines.append(heading + "\n")
                experience = (pr.get("experience") or "").strip()
                if experience:
                    lines.append(f"{experience}\n")
                friction = pr.get("friction") or []
                for point in friction:
                    point = (point or "").strip()
                    if point:
                        lines.append(f"- {point}")
                if friction:
                    lines.append("")  # blank line after the bullet list
            if overall:
                lines.append("## Overall\n")
                lines.append(f"{overall}\n")

        related = report.get("related_steps") or []
        if related:
            lines.append("## Related steps\n")
            lines.append(", ".join(f"step {s}" for s in related) + "\n")

        error_text = report.get("error_text")
        source = report.get("error_source_file")
        if error_text or source:
            lines.append("## Error\n")
            if source:
                lines.append(f"**Source file:** `{source}`\n")
            if error_text:
                lines.append("```\n" + error_text.rstrip() + "\n```\n")

        meta = f"_status: {report.get('status', 'unknown')} · severity: {report.get('severity', 'info')}_\n"
        lines.append(meta)

        run_dir = self.runs_dir / self.run_id
        os.makedirs(run_dir, exist_ok=True)
        path = run_dir / "NEXT_ACTION.md"
        try:
            path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("📝 Wrote next action: %s", path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"⚠️  Failed to write NEXT_ACTION.md: {exc}")