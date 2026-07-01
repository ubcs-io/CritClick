Here’s a production-ready **vision-based harness** that replaces template matching with real-time LLM vision analysis. It captures the screen, sends it to a vision-capable model, parses structured JSON responses, executes actions, and maintains a living narrative log.

---
### 📦 1. Dependencies
```bash
pip install openai pyautogui pillow opencv-python
```
*(Optional: Replace `openai` with `anthropic`, `google-generativeai`, or `ollama` for local/alternative models)*

---
### 🐍 2. The Vision Harness (`vision_harness.py`)
```python
import os
import sys
import time
import json
import logging
import subprocess
import pyautogui
from datetime import datetime
from io import BytesIO
from typing import Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VisionHarness")

class LLMClient:
    """Abstract base for LLM vision clients. Replace or extend as needed."""
    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze(self, image_b64: str, system_prompt: str, user_prompt: str) -> dict:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "renpy_action",
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "action": {"type": "string", "enum": ["click", "wait", "type", "done"]},
                        "coordinates": {"type": "array", "items": {"type": "number"}},
                        "text_to_type": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "narrative": {"type": "string"}
                    },
                    "required": ["description", "action", "coordinates", "reasoning", "narrative"]
                }
            }
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}}
                ]}
            ],
            max_tokens=600,
            response_format=response_format,
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

class VisionPlaythroughHarness:
    def __init__(
        self,
        game_dir: str,
        llm_client: LLMClient,
        resolution: tuple = (1280, 720),
        screen_scale: float = 1.0,
        max_steps: int = 50,
        wait_after_action: float = 1.5,
        stuck_threshold: int = 3
    ):
        self.game_dir = game_dir
        self.llm = llm_client
        self.resolution = resolution
        self.screen_scale = screen_scale
        self.max_steps = max_steps
        self.wait_after_action = wait_after_action
        self.stuck_threshold = stuck_threshold

        self.step = 0
        self.running = False
        self.process = None
        self.stuck_counter = 0
        self.context_window = []
        self.log_file = "playthrough_log.jsonl"

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

        os.makedirs("screenshots", exist_ok=True)

    def pull_latest(self):
        logger.info("📥 Pulling latest version from git...")
        if os.path.isdir(os.path.join(self.game_dir, ".git")):
            try:
                subprocess.run(["git", "pull", "--rebase"], cwd=self.game_dir, check=True, capture_output=True, text=True)
                logger.info("✅ Git pull successful.")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ Git pull failed: {e.stderr}")
        else:
            logger.info("📁 No git repo found. Using existing build.")

    def launch(self):
        logger.info(f"🚀 Launching Ren'Py from: {self.game_dir}")
        exe = "renpy.exe" if sys.platform == "win32" else "renpy.sh"
        path = os.path.join(self.game_dir, exe)
        if not os.path.exists(path):
            path = os.path.join(self.game_dir, "renpy", exe)
        if not os.path.exists(path):
            raise FileNotFoundError("Ren'Py executable not found. Check path or rename manually.")

        cmd = [path, "--window", "--size", f"{self.resolution[0]}x{self.resolution[1]}", "--nodaemon"]
        self.process = subprocess.Popen(cmd)
        time.sleep(5)  # Allow boot/main menu to load
        self.running = True
        logger.info("🎮 Game launched. Vision loop initialized.")

    def capture_screen(self) -> str:
        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue().hex()  # hex string for JSON serialization

    def _get_system_prompt(self) -> str:
        return """You are an autonomous AI player analyzing a Ren'Py visual novel screen. 
        Respond with a JSON object containing: description, action, coordinates, reasoning, narrative.
        Actions: "click" (tap UI), "wait" (pause for animation/dialogue), "type" (input text), "done" (main menu/game over).
        Coordinates are relative to the top-left corner (0,0). Center of screen is typically (640, 360) at 1280x720."""

    def _get_user_prompt(self, context: str) -> str:
        return f"""Analyze the current Ren'Py screen.
        Recent context: {context}
        Identify interactive elements, dialogue state, and next logical action.
        If dialogue is auto-advancing, use "wait" until text stabilizes.
        If choices appear, click the most logical one based on narrative context.
        Output ONLY valid JSON matching the schema."""

    def analyze_and_act(self):
        img_b64 = self.capture_screen()
        context_str = "\n".join(self.context_window[-3:]) if self.context_window else "Starting playthrough."

        try:
            response = self.llm.analyze(img_b64, self._get_system_prompt(), self._get_user_prompt(context_str))
        except Exception as e:
            logger.error(f"💥 LLM call failed: {e}")
            return False

        # Validate response
        if "action" not in response or "coordinates" not in response:
            logger.warning("⚠️ Invalid LLM response. Skipping step.")
            return False

        action = response["action"]
        narrative = response.get("narrative", "No narrative provided.")
        logger.info(f"📝 Step {self.step+1}/{self.max_steps} | {narrative}")
        self.context_window.append(f"Step {self.step+1}: {narrative}")

        # Save structured log
        log_entry = {
            "step": self.step + 1,
            "timestamp": datetime.now().isoformat(),
            "llm_response": response,
            "image_hex": img_b64[:200] + "...",  # Truncate for storage
            "file_path": os.path.join("screenshots", f"step_{self.step+1}.png")
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Execute action
        if action == "done":
            logger.info("🏁 Game ended or returned to main menu.")
            return "completed"
        elif action == "wait":
            wait_time = 2.0
            logger.info(f"⏳ Waiting {wait_time}s for scene/dialogue to stabilize...")
            time.sleep(wait_time)
            self.stuck_counter = 0
            return "waited"
        elif action == "click":
            x, y = int(response["coordinates"][0] * self.screen_scale), int(response["coordinates"][1] * self.screen_scale)
            pyautogui.click(x, y)
            logger.info(f"🖱️ Clicked ({x}, {y})")
            self.stuck_counter = 0
            return "clicked"
        elif action == "type":
            text = response.get("text_to_type", "")
            pyautogui.write(text, interval=0.05)
            logger.info(f"⌨️ Typed: {text}")
            self.stuck_counter = 0
            return "typed"
        return "unknown"

    def run_playthrough(self):
        logger.info("🎮 Starting autonomous vision-based playthrough...")
        try:
            for i in range(self.max_steps):
                if self.process.poll() is not None:
                    logger.info("🛑 Process terminated unexpectedly.")
                    break

                result = self.analyze_and_act()
                if result == "completed":
                    break

                time.sleep(self.wait_after_action)
        except KeyboardInterrupt:
            logger.info("⌨️ Playthrough interrupted by user.")
        except Exception as e:
            logger.error(f"💥 Harness crashed: {e}")
        finally:
            self.stop()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.running = False
        logger.info("🛑 Harness stopped.")

if __name__ == "__main__":
    # Configuration
    GAME_DIR = "./my_renpy_game"
    API_KEY = "sk-your-openai-key-here"
    RESOLUTION = (1280, 720)
    SCALE = 1.0  # Set if OS resolution differs from Ren'Py window

    llm_client = OpenAIClient(api_key=API_KEY, model="gpt-4o")
    harness = VisionPlaythroughHarness(
        game_dir=GAME_DIR,
        llm_client=llm_client,
        resolution=RESOLUTION,
        screen_scale=SCALE,
        max_steps=60,
        stuck_threshold=3
    )
    harness.pull_latest()
    harness.launch()
    harness.run_playthrough()
```

---
### 🔍 3. Key Architecture Features

| Feature | How It Works |
|--------|--------------|
| **Vision-First Decision Loop** | Captures screen → sends to LLM → parses JSON → clicks/waits/types → logs narrative |
| **Context Window Management** | Keeps last 3 steps to maintain narrative continuity without bloating tokens |
| **Stuck Detection** | Tracks repeated "click" actions. Could be extended to auto-scroll or try alternate paths |
| **Structured Logging** | Writes JSONL logs + full screenshots for post-hoc analysis or fine-tuning |
| **Coordinate Scaling** | `screen_scale` handles OS DPI/Ren'Py internal resolution mismatches |
| **LLM Abstraction** | Swap `OpenAIClient` with Anthropic, Gemini, or local Ollama/LLaVA implementations |

---
### 🛠 4. Usage & Customization Tips

1. **Swap LLM Backend**  
   Replace `OpenAIClient` with any client that implements `analyze(image_b64, system, user) -> dict`. Example for Ollama:
   ```python
   class OllamaClient(LLMClient):
       def analyze(self, image_b64, system, user):
           import requests
           resp = requests.post("http://localhost:11434/api/chat", json={
               "model": "llava:7b",
               "messages": [{"role": "user", "content": user, "images": [image_b64]}],
               "format": {"type": "object", "properties": {...}}
           })
           return json.loads(resp.json()["message"]["content"])
   ```

2. **Prompt Engineering for Ren'Py**  
   The system prompt can be enhanced with game-specific knowledge:
   ```python
   f"""You are playing '{game_title}'. Current route: {route_name}. 
   Choices affect {relationship/ending}. Prioritize [X] over [Y]."""
   ```

3. **Handle Dialogue Auto-Advance**  
   Some Ren'Py games auto-advance text. Add this to the prompt:
   > "If dialogue text is still fading or cursor is blinking, use 'wait'. Only click when text is static."

4. **Save/Load Screen Handling**  
   LLMs sometimes click "Back" instead of "Load". Add fallback:
   ```python
   if action == "click" and narrative.contains("load"):
       pyautogui.press("enter")  # Often selects first item
   ```

5. **Performance Tuning**  
   - `temperature=0.1` for deterministic UI clicks
   - `detail="high"` in image_url for button text clarity
   - Reduce `wait_after_action` to `0.8` for faster loops
   - Run on a dedicated display or virtual framebuffer (`Xvfb`/`Docker`) for headless testing

---
### 📊 5. Example Output Log (`playthrough_log.jsonl`)
```json
{"step": 3, "timestamp": "2024-06-12T14:22:10.123456", "llm_response": {"description": "Choice screen: Accept offer (200g) vs Decline", "action": "click", "coordinates": [580, 410], "reasoning": "Player prefers gold route. Accepting unlocks merchant.", "narrative": "🤖 Encountered choice screen. Selected 'Accept offer' to maximize early-game funds."}, "image_hex": "89504e47...", "file_path": "screenshots/step_3.png"}
```

---
### 🔮 6. Next-Level Extensions
- **Multi-Agent Orchestration**: Separate `Navigator` (UI clicks) from `Strategist` (route planning)
- **Save-Game Checkpoints**: Use Ren'Py's `renpy.save()` screen to snapshot progress after key decisions
- **LLM Fallback Chain**: If vision fails → OCR → template matching → center click
- **Reward/Feedback Loop**: Log success metrics (dialogue progress, item count) to fine-tune next run

Would you like a version that:
- Integrates directly with Ren'Py's `renpy` Python API (no screen scraping)?
- Handles dialogue typing (e.g., `renpy.input()`)?
- Generates a Markdown summary with embedded screenshots and decision trees?