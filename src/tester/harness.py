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
from .models import ActionResponse, LogEntry
from .prompts import make_system_prompt, make_user_prompt
from .screen import Capturer, ScreenCaptureError

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
    ) -> None:
        self.settings = settings
        self.launcher = launcher
        self.llm = llm_client
        self.dry_run = dry_run
        self.capturer = Capturer(scale=settings.harness.screen_scale)

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
            self._write_manifest()

        return exit_code

    def stop(self) -> None:
        """Clean shutdown — stop the game, finalise logging."""
        if not self.dry_run:
            self.launcher.stop()
        logger.info("🛑 Harness stopped.")

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
            logger.error(f"💥 Screen capture failed: {exc}")
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
            logger.error(f"💥 LLM call failed: {exc}")
            return "unknown"
        duration_ms = int((time.time() - llm_start) * 1000)

        # 4. Validate with Pydantic
        try:
            response = ActionResponse.model_validate(raw)
        except Exception as exc:
            logger.warning(f"⚠️  Invalid LLM response (schema): {exc}")
            return "unknown"

        # 5. Log narrative
        logger.info(
            "📝 Step %d/%d | %s",
            self.step,
            self.settings.harness.max_steps,
            response.narrative,
        )
        self.context_window.append(f"Step {self.step}: {response.narrative}")
        self._write_log_entry(response, image_b64, duration_ms)

        # 6. Execute
        return self._execute(response)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, response: ActionResponse) -> str:
        action = response.action.value

        # Stuck detection
        if action == self.last_action:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_action = action

        if self.stuck_counter >= self.settings.harness.stuck_threshold:
            logger.warning(
                "⚠️  Possible stuck state — %d identical '%s' actions.",
                self.stuck_counter,
                action,
            )

        # Track action counts for summary
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        if action == "done":
            logger.info("🏁 Game ended or returned to main menu.")
            return "completed"

        if action == "wait":
            wait_time = self.settings.harness.wait_duration
            logger.info(f"⏳ Waiting {wait_time}s for scene to stabilise…")
            if not self.dry_run:
                time.sleep(wait_time)
            return "waited"

        if action == "click":
            if len(response.coordinates) < 2:
                logger.warning("⚠️  'click' action without valid coordinates — skipping.")
                return "unknown"

            x, y = self.capturer.scale_coordinates(
                response.coordinates[0], response.coordinates[1]
            )

            # Bounds validation
            if not self._validate_coordinates(x, y):
                logger.warning(
                    "⚠️  Click coordinates (%d, %d) are outside expected window bounds — skipping.",
                    x, y,
                )
                return "unknown"

            if not self.dry_run:
                pyautogui.click(x, y)
                logger.info(f"🖱️  Clicked ({x}, {y})")
            else:
                logger.info(f"🔍 [DRY RUN] Would click ({x}, {y})")
            return "clicked"

        if action == "type":
            text = response.text_to_type or ""
            if not self.dry_run:
                pyautogui.write(text, interval=0.05)
                logger.info(f"⌨️  Typed: {text}")
            else:
                logger.info(f"🔍 [DRY RUN] Would type: {text}")
            return "typed"

        if action == "press":
            key = response.key_to_press or ""
            if not key:
                logger.warning("⚠️  'press' action without key_to_press — skipping.")
                return "unknown"
            if not self.dry_run:
                pyautogui.press(key)
                logger.info(f"⌨️  Pressed key: {key}")
            else:
                logger.info(f"🔍 [DRY RUN] Would press key: {key}")
            return "pressed"

        logger.warning(f"⚠️  Unknown action: {action}")
        return "unknown"

    def _validate_coordinates(self, x: int, y: int) -> bool:
        """Check whether coordinates fall within the expected game window bounds."""
        w, h = self.settings.game.resolution
        # Allow a small margin for window borders/title bars
        margin = 50
        return (-margin <= x <= w + margin) and (-margin <= y <= h + margin)

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
            filename = f"step_{self.step:04d}.png"
            screenshot_path = os.path.join(log_settings.screenshot_dir, filename)
            try:
                img = self.capturer.capture_pil()
                img.save(screenshot_path)
            except Exception as exc:
                logger.warning(f"⚠️  Failed to save screenshot: {exc}")
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

    def _write_manifest(self) -> None:
        """Write a ``run_manifest.json`` describing this run.

        Only written when ``run_id`` is set (namespaced mode). The manifest is
        the canonical entry point for the ingestion system: discover runs by
        globbing ``runs/*/run_manifest.json``.
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
