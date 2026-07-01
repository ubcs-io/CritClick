# tester — Vision-Based Autonomous Game Testing Harness

**tester** is a vision-driven game testing framework. It captures your game's screen, sends it to any **OpenAI-compatible vision LLM endpoint**, parses structured JSON responses, executes mouse/keyboard actions, and maintains a structured playthrough log.

Supports **Ren'Py**, **Godot**, and **custom executables** out of the box.

---

## 🚀 Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Create a config

```bash
tester --new-config > my_game.toml
# Edit my_game.toml with your game path, LLM endpoint, and API key
```

### 3. Run

```bash
tester --config my_game.toml
```

---

## 🔧 Configuration

### By TOML file

```toml
[llm]
api_base = "https://api.openai.com/v1"   # Any OpenAI-compatible endpoint
api_key = "sk-..."                         # Or set TESTER_API_KEY env var
model = "gpt-4o"

[game]
type = "renpy"                             # "renpy", "godot", or "custom"
path = "./my_renpy_game"
resolution = [1280, 720]

[harness]
max_steps = 60
```

### By environment variables

All settings can be overridden with `TESTER_*` environment variables:

```bash
export TESTER_LLM__API_BASE="http://localhost:8000/v1"
export TESTER_LLM__MODEL="llava-v1.6-mistral-7b"
export TESTER_GAME__TYPE="godot"
export TESTER_GAME__PATH="./my_godot_game"
export TESTER_GAME__RESOLUTION='[1280, 720]'

tester   # No --config needed
```

---

## 🔌 Any OpenAI-Compatible Endpoint

Set `api_base` to any of these:

| Service | `api_base` |
|---------|-----------|
| **OpenAI** | `https://api.openai.com/v1` |
| **Azure OpenAI** | `https://<resource>.openai.azure.com` |
| **vLLM** | `http://localhost:8000/v1` |
| **Ollama** (OpenAI compat) | `http://localhost:11434/v1` |
| **LiteLLM** | `http://localhost:4000` |
| **Any local proxy** | `http://localhost:8080/v1` |

---

## 🎮 Supported Game Engines

| Engine | Config `type` | Auto-detection |
|--------|--------------|----------------|
| **Ren'Py** | `renpy` | Looks for `renpy.sh`/`renpy.exe` in the game directory |
| **Godot** | `godot` | Looks for `project.godot` or exported binary |
| **Custom** | `custom` | Runs any executable with optional args |

### Example: Ren'Py

```toml
[game]
type = "renpy"
path = "./my_visual_novel"
git_pull = true          # Auto pull latest from git before starting
```

### Example: Godot

```toml
[game]
type = "godot"
path = "./my_game/project.godot"
executable = "godot"     # Godot editor binary
```

---

## 📊 Output

All playthrough data is logged to a JSONL file (default: `playthrough_log.jsonl`):

```jsonl
{"step":1, "timestamp":"2024-06-12T14:22:10", "llm_response":{"description":"Title screen with 'Start' button", "action":"click", "coordinates":[640, 450], "reasoning":"Click Start to begin the game.", "narrative":"🎬 Title screen detected. Starting new game."}, "image_hex_truncated":"iVBORw0KGgo...", "screenshot_path":"screenshots/step_0001.png"}
```

Screenshots are saved per-step when `save_screenshots = true`.

---

## 🧪 Command Reference

```bash
tester --config path/to/config.toml     # Run a playthrough
tester --new-config                     # Print a default config file
tester --list-games ./game_dir          # Scan for Ren'Py/Godot projects
tester --version                        # Show version
tester --verbose --config ...            # Debug logging
```

---

## 🏗️ Project Structure

```
src/tester/
├── __init__.py     # Package metadata
├── __main__.py     # python -m tester entry
├── cli.py          # CLI argument parsing
├── config.py       # Settings (TOML + env vars)
├── client.py       # OpenAI-compatible LLM client
├── harness.py      # Main playthrough loop
├── launcher.py     # Game process launcher
├── models.py       # Pydantic data models
├── prompts.py      # Prompt templates
└── screen.py       # Screen capture
```

---

## 🔮 Extending

- **Add a game engine**: Subclass `Launcher` and register it in `launcher._LAUNCHER_MAP`
- **Swap the LLM client**: Implement `LLMClient` (e.g., Anthropic, Gemini, local model)
- **Custom prompts**: Set `system_prompt` / `user_prompt_template` in your TOML config
- **Headless CI**: Install `python-xlib` and use `headless = true` with Xvfb on Linux

---

## 📦 Dependencies

- `openai>=1.0` — LLM client
- `pydantic>=2.0` — Config & response models
- `mss` — Fast cross-platform screen capture
- `pillow` — Image handling
- `pyautogui` — Mouse/keyboard automation
