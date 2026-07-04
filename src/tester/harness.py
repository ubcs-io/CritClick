"""Core vision-based playthrough harness.

Orchestrates the capture → analyse → act → log loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pyautogui

from .client import LLMClient
from .config import Settings
from .launcher import Launcher, capture_git_info
from .models import ActionResponse, LogEntry, RecapReport
from .prompts import (
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
    ) -> None:
        self.settings = settings
        self.launcher = launcher
        self.llm = llm_client
        self.dry_run = dry_run
        self.debug_harness = debug_harness
        self.debug_screen = debug_screen
        self.capturer = Capturer(
            scale=settings.harness.screen_scale,
            debug=debug_screen,
            game_resolution=tuple(settings.game.resolution),
        )

        # Run namespacing — when run_id is set, outputs go under
        # ``<runs_dir>/<run_id>/`` instead of the flat cwd paths.
        # When None, behaviour is unchanged (backward compatible).
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)

        # Runtime state
        self.step = 0
        self.context_window: list[str] = []
        self.stuck_counter = 0
        self.last_action: str | None = None
        self.start_time: float | None = None
        self.completion_reason: str | None = None
        self.action_counts: dict[str, int] = {}
        self._recovery_attempts = 0
        self._recovery_strategies: list[str] = []
        self._window_tracker: WindowTracker | None = None

        # Captured at run start, written into the manifest
        self._git_info: dict | None = None
        self._exit_code: int | None = None

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
        """Create screenshot dir and ensure the log file path is ready."""
        log_settings = self.settings.logging
        if log_settings.save_screenshots:
            os.makedirs(log_settings.screenshot_dir, exist_ok=True)
        # Touch the log file
        Path(log_settings.log_file).parent.mkdir(parents=True, exist_ok=True)

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

        exit_code = EXIT_SUCCESS
        try:
            for step_num in range(1, self.settings.harness.max_steps + 1):
                self.step = step_num
                if not self.dry_run and not self.launcher.is_alive():
                    logger.warning("⚠️  Game process died unexpectedly.")
                    self.completion_reason = "game_died"
                    exit_code = EXIT_GAME_DIED
                    self._log_game_output()
                    break

                result = self.analyze_and_act()
                if result == "completed":
                    logger.info("🏁 Playthrough completed naturally.")
                    self.completion_reason = "completed"
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
        finally:
            self.stop()
            self._exit_code = exit_code
            self._log_summary()
            recap = self._run_recap()
            self._write_manifest(recap=recap)

        return exit_code

    def stop(self) -> None:
        """Clean shutdown — stop the game, finalise logging."""
        if not self.dry_run:
            self.launcher.stop()
        logger.info("🛑 Harness stopped.")

    def _log_game_output(self) -> None:
        """Read and log any stdout/stderr produced by the game process.

        Called when the game process dies unexpectedly so that error messages
        that would otherwise be swallowed are surfaced in the log.
        """
        process = self.launcher.process
        if process is None:
            return

        # If the process hasn't been reaped yet, read the captured pipes
        stdout, stderr = process.communicate()
        if stdout:
            try:
                stdout_text = stdout.decode("utf-8", errors="replace")
            except Exception:
                stdout_text = str(stdout)
            if stdout_text.strip():
                logger.info(
                    "📤 Game stdout:\n%s",
                    stdout_text[-4000:],
                )
        if stderr:
            try:
                stderr_text = stderr.decode("utf-8", errors="replace")
            except Exception:
                stderr_text = str(stderr)
            if stderr_text.strip():
                logger.info(
                    "📥 Game stderr:\n%s",
                    stderr_text[-4000:],
                )

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
            image_b64 = self.capturer.capture_base64()
        except ScreenCaptureError as exc:
            logger.error("Step %d/%d | 💥 Screen capture failed: %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"

        # 2. Build prompts
        context_str = (
            "\n".join(self.context_window[-self.settings.logging.context_window_size :])
            if self.context_window
            else "Starting playthrough."
        )
        system_prompt = make_system_prompt(self.settings.system_prompt)
        user_prompt = make_user_prompt(context_str, self.settings.user_prompt_template)

        # 3. Analyse via LLM (timed)
        llm_start = time.time()
        try:
            raw = self.llm.analyze(image_b64, system_prompt, user_prompt)
        except RuntimeError as exc:
            logger.error("Step %d/%d | 💥 LLM call failed: %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"
        duration_ms = int((time.time() - llm_start) * 1000)

        # 4. Validate with Pydantic
        try:
            response = ActionResponse.model_validate(raw)
        except Exception as exc:
            logger.warning("Step %d/%d | ⚠️  Invalid LLM response (schema): %s", self.step, self.settings.harness.max_steps, exc)
            return "unknown"

        # 5. Debug: log full parsed ActionResponse
        if self.debug_harness:
            logger.info(
                "🔍 [DEBUG-HARNESS] Step %d/%d | ActionResponse: action=%s | coords=%s | text=%s | key=%s | narrative=%s | reasoning=%s",
                self.step,
                self.settings.harness.max_steps,
                response.action.value,
                response.coordinates,
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
        self.context_window.append(f"Step {self.step}: {response.narrative}")
        self._write_log_entry(response, image_b64, duration_ms)

        # 7. Execute
        return self._execute(response)

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
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'wait' → sleeping %.1fs",
                    self.step, self.settings.harness.max_steps, wait_time,
                )
            logger.info("Step %d/%d | ⏳ Waiting %ss for scene to stabilise…", self.step, self.settings.harness.max_steps, wait_time)
            if not self.dry_run:
                time.sleep(wait_time)
            return "waited"

        if action == "click":
            # Resolve click target: prefer bounding_box centre, fall back to coordinates
            bb = response.bounding_box
            if bb is not None and len(bb) == 4:
                # Centre of bounding box: [x1, y1, x2, y2]
                raw_x = (bb[0] + bb[2]) / 2.0
                raw_y = (bb[1] + bb[3]) / 2.0
                target_source = f"b-box centre from [{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]"
            elif len(response.coordinates) >= 2:
                raw_x = response.coordinates[0]
                raw_y = response.coordinates[1]
                target_source = "raw coordinates"
            else:
                logger.warning("Step %d/%d | ⚠️  'click' action without valid coordinates or bounding_box — skipping.", self.step, self.settings.harness.max_steps)
                return "unknown"

            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Action 'click' | target=%s | raw=(%.1f, %.1f) | capturer.scale=%.4f",
                    self.step, self.settings.harness.max_steps,
                    target_source, raw_x, raw_y, self.capturer.scale,
                )

            x, y = self.capturer.scale_coordinates(raw_x, raw_y)

            if self.debug_harness:
                screen_w, screen_h = pyautogui.size()
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Scaled coords: (%d, %d) | screen: %dx%d",
                    self.step, self.settings.harness.max_steps, x, y, screen_w, screen_h,
                )

            # Bounds validation
            if not self._validate_coordinates(x, y):
                screen_w, screen_h = pyautogui.size()
                if self.debug_harness:
                    logger.warning(
                        "🔍 [DEBUG-HARNESS] Step %d/%d | BOUNDS CHECK FAILED: (%d, %d) outside screen %dx%d (±50 margin)",
                        self.step, self.settings.harness.max_steps, x, y, screen_w, screen_h,
                    )
                logger.warning(
                    "Step %d/%d | ⚠️  Click coordinates (%d, %d) are outside expected screen bounds — skipping.",
                    self.step,
                    self.settings.harness.max_steps,
                    x, y,
                )
                return "unknown"

            if self.debug_harness:
                logger.info(
                    "🔍 [DEBUG-HARNESS] Step %d/%d | Bounds check passed — proceeding with click (%s)",
                    self.step, self.settings.harness.max_steps, target_source,
                )

            if not self.dry_run:
                self.capturer.click(x, y)
                logger.info("Step %d/%d | 🖱️  Clicked (%d, %d) [scale=%.2f, %s]", self.step, self.settings.harness.max_steps, x, y, self.capturer.scale, target_source)
            else:
                logger.info("Step %d/%d | 🔍 [DRY RUN] Would click (%d, %d) [%s]", self.step, self.settings.harness.max_steps, x, y, target_source)
            # Feed the actual click coordinates back to the LLM context
            # so it can self-correct if clicks are landing in the wrong place.
            self.context_window.append(
                f"Click executed at absolute screen coordinates ({x}, {y}) [{target_source}]."
            )
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
            if self.run_id:
                run_prefix = self.run_id[:5]
                filename = f"{run_prefix}_step_{self.step:04d}.png"
            else:
                filename = f"step_{self.step:04d}.png"
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

    def _log_summary(self) -> None:
        """Log a run summary at the end of the playthrough."""
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
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, recap: dict | None = None) -> None:
        """Write a ``run_manifest.json`` describing this run.

        Only written when ``run_id`` is set (namespaced mode). The manifest is
        the canonical entry point for the ingestion system: discover runs by
        globbing ``runs/*/run_manifest.json``.

        Args:
            recap: Optional recap report dict (e.g. ``{"key_complaint": "..."}``)
                   to embed in the manifest.
        """
        if not self.run_id:
            return

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

        if recap:
            manifest["recap"] = recap

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
    # Run recap
    # ------------------------------------------------------------------

    def _run_recap(self) -> dict | None:
        """Send all step logs to the LLM for a post-run recap.

        Reads back the JSONL log file, formats a recap prompt, and calls the
        LLM to extract the key complaint from the playthrough.

        Returns:
            A dict with a ``"key_complaint"`` field, or ``None`` if the recap
            could not be generated (missing log file, LLM error, etc.).
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
            duration = entry.get("duration_ms", 0)
            step_lines.append(
                f"Step {step_num} [{action}] ({duration}ms): {narrative}\n"
                f"  Reasoning: {reasoning}"
            )

        step_log = "\n".join(step_lines)

        # Build recap prompts
        duration = time.time() - self.start_time if self.start_time else 0
        actions_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.action_counts.items())
        ) or "none"
        system_prompt = make_recap_system_prompt()
        user_prompt = make_recap_user_prompt(
            steps_completed=self.step,
            completion_reason=self.completion_reason or "unknown",
            duration=duration,
            action_counts=actions_str,
            step_log=step_log,
        )

        # Call LLM — recap is text-only, so send a minimal 1×1 PNG placeholder
        # to satisfy the vision API contract while keeping focus on the text.
        PLACEHOLDER_PNG = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        try:
            raw = self.llm.analyze(PLACEHOLDER_PNG, system_prompt, user_prompt)
        except RuntimeError as exc:
            logger.warning("📋 Recap LLM call failed: %s", exc)
            return None

        try:
            recap = RecapReport.model_validate(raw)
        except Exception as exc:
            logger.warning("📋 Recap response validation failed: %s", exc)
            return None

        recap_dict = recap.model_dump()
        logger.info("📋 Run recap | key_complaint: %s", recap_dict["key_complaint"])
        return recap_dict
