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
from typing import Optional

import pyautogui

from .client import LLMClient
from .config import Settings
from .launcher import Launcher
from .models import ActionResponse, LogEntry
from .prompts import make_system_prompt, make_user_prompt
from .screen import Capturer, ScreenCaptureError

logger = logging.getLogger("tester.harness")


class Harness:
    """Vision-based playthrough harness.

    The main loop:
      1. Capture screen → base64
      2. Send to LLM → structured JSON
      3. Validate & execute action (click / wait / type / done)
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
    ) -> None:
        self.settings = settings
        self.launcher = launcher
        self.llm = llm_client
        self.capturer = Capturer(scale=settings.harness.screen_scale)

        # Runtime state
        self.step = 0
        self.context_window: list[str] = []
        self.stuck_counter = 0
        self.last_action: Optional[str] = None

        # Setup output directories
        self._setup_output()

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

        logger.info(
            "🎮 Harness initialised | max_steps=%d | model=%s | endpoint=%s",
            settings.harness.max_steps,
            settings.llm.model,
            settings.llm.api_base,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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

    def run(self) -> None:
        """Run the full playthrough loop."""
        logger.info("🎮 Starting autonomous vision-based playthrough…")

        # Optional git pull
        self.launcher.pull_latest()

        # Launch the game
        self.launcher.launch()

        try:
            for self.step in range(1, self.settings.harness.max_steps + 1):
                if not self.launcher.is_alive():
                    logger.warning("⚠️  Game process died unexpectedly.")
                    break

                result = self.analyze_and_act()
                if result == "completed":
                    logger.info("🏁 Playthrough completed naturally.")
                    break

                time.sleep(self.settings.harness.wait_after_action)

        except KeyboardInterrupt:
            logger.info("⌨️ Playthrough interrupted by user.")
        except Exception as exc:
            logger.exception(f"💥 Harness crashed: {exc}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Clean shutdown — stop the game, finalise logging."""
        self.launcher.stop()
        logger.info("🛑 Harness stopped.")

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def analyze_and_act(self) -> str:
        """Single capture → analyse → act → log cycle.

        Returns:
            One of ``"completed"``, ``"clicked"``, ``"waited"``, ``"typed"``,
            or ``"unknown"``.
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

        # 3. Analyse via LLM
        try:
            raw = self.llm.analyze(image_b64, system_prompt, user_prompt)
        except RuntimeError as exc:
            logger.error(f"💥 LLM call failed: {exc}")
            return "unknown"

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
        self._write_log_entry(response, image_b64)

        # 6. Execute
        return self._execute(response)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, response: ActionResponse) -> str:
        action = response.action.value
        narrative = response.narrative

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

        if action == "done":
            logger.info("🏁 Game ended or returned to main menu.")
            return "completed"

        if action == "wait":
            wait_time = 2.0
            logger.info(f"⏳ Waiting {wait_time}s for scene to stabilise…")
            time.sleep(wait_time)
            self.stuck_counter = 0
            return "waited"

        if action == "click":
            if len(response.coordinates) < 2:
                logger.warning("⚠️  'click' action without valid coordinates — skipping.")
                return "unknown"
            x, y = self.capturer.scale_coordinates(
                response.coordinates[0], response.coordinates[1]
            )
            pyautogui.click(x, y)
            logger.info(f"🖱️  Clicked ({x}, {y})")
            self.stuck_counter = 0
            return "clicked"

        if action == "type":
            text = response.text_to_type or ""
            pyautogui.write(text, interval=0.05)
            logger.info(f"⌨️  Typed: {text}")
            self.stuck_counter = 0
            return "typed"

        logger.warning(f"⚠️  Unknown action: {action}")
        return "unknown"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log_entry(self, response: ActionResponse, image_b64: str) -> None:
        """Append a structured log entry to the JSONL file and optionally save a screenshot."""
        log_settings = self.settings.logging
        screenshot_path: Optional[str] = None

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
            image_hex_truncated=image_b64[:200] + "...",
            screenshot_path=screenshot_path,
        )

        with open(log_settings.log_file, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
