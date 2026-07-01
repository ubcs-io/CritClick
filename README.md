# tester — Vision-Based Autonomous Game Testing Harness

**tester** is a vision-driven game testing framework. It captures your game's screen, sends it to any **OpenAI-compatible vision LLM endpoint**, parses structured JSON responses, executes mouse/keyboard actions, and maintains a structured playthrough log — with retries, safety guards, and report generation.

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

### 3. Validate before running

```bash
tester --check --config my_game.toml
```

### 4. Run

```bash
tester --config my_game.toml
```

### 5. Generate a report

```bash
tester --report playthrough_log.jsonl
```

---

## 🧪 Command Reference

```bash
tester --config path/to/config.toml     # Run a playthrough
tester --new-config                     # Print a default config file
tester --list-games ./game_dir          # Scan for Ren'Py/Godot projects
tester --check --config ...             # Validate config + paths + API connectivity
tester --dry-run --config ...           # Analyse without launching or acting
tester --report playthrough_log.jsonl   # Convert log to Markdown report
tester --version                        # Show version
tester --verbose --config ...           # Debug logging
tester --config ... --run-id <id>       # Namespace outputs under runs/<id>/
tester --config ... --lock-file <path>  # Prevent overlapping runs
```

---

## 🕒 Scheduled / Autonomous Deployment

`tester` is decoupled from game code — it launches the game as a subprocess
by path and never imports it. The package itself is **one-shot** (one
playthrough per invocation), designed to be triggered by an **external
scheduler** (systemd, cron, launchd).

**Per-run namespacing** (`--run-id`) writes each playthrough to its own
directory with a machine-readable manifest for ingestion:

```
runs/20260701T165500Z/
├── run_manifest.json       # ← canonical entry point for ingestion
├── playthrough_log.jsonl
└── screenshots/
```

Discover runs by globbing `runs/*/run_manifest.json`. The manifest includes
the game's **git commit**, so playthrough quality can be correlated to code
changes over time.

**Lock file** (`--lock-file`) uses `fcntl.flock` to prevent overlapping runs
from fighting over a shared display — if a run overruns its slot, the next
invocation exits cleanly with code 4 (`EXIT_LOCKED`).

For a complete headless-Linux deployment guide with systemd timers and Xvfb,
see **[deploy/README.md](deploy/README.md)**.

---

## 🔧 Configuration

### By TOML file

```toml
[llm]
api_base = "https://api.openai.com/v1"   # Any OpenAI-compatible endpoint
api_key = "sk-..."                         # Or set OPENAI_API_KEY env var
model = "gpt-4o"
max_retries = 3                            # Retry transient failures
retry_delay = 2.0                          # Base delay for exponential backoff

[game]
type = "renpy"                             # "renpy", "godot", or "custom"
path = "./my_renpy_game"
resolution = [1280, 720]

[harness]
max_steps = 50
wait_duration = 2.0                        # Seconds to pause on 'wait' actions
startup_countdown = true                   # 3-second countdown with failsafe reminder
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

## 🎯 Available Actions

The LLM can choose from five action types:

| Action | Description | Fields |
|--------|-------------|--------|
| `click` | Click at (x, y) coordinates | `coordinates` |
| `wait` | Pause for scene transitions/animations | — |
| `type` | Type text into an input field | `text_to_type` |
| `press` | Press a single key (Enter, Escape, etc.) | `key_to_press` |
| `done` | Game has reached a natural end point | — |

---

## 🛡️ Safety Features

- **Startup countdown** — A 3-second countdown with failsafe reminder before the game launches. Disable with `[harness].startup_countdown = false`.
- **Coordinate bounds validation** — Click coordinates outside the expected game window are rejected, preventing accidental clicks on other applications.
- **Dry-run mode** — Run `tester --dry-run` to exercise the full capture → analyse → log loop without launching the game or executing any actions.
- **PyAutoGUI failsafe** — Move mouse to any screen corner to abort immediately.
- **Exit codes** — `0` (success), `1` (config error), `2` (runtime error), `3` (game died), `4` (locked), `130` (interrupted).

---

##  Output

### JSONL Log

All playthrough data is logged to a JSONL file (default: `playthrough_log.jsonl`):

```jsonl
{"step":1, "timestamp":"2024-06-12T14:22:10", "duration_ms":1234, "llm_response":{"description":"Title screen with 'Start' button", "action":"click", "coordinates":[640, 450], "reasoning":"Click Start to begin the game.", "narrative":"🎬 Title screen detected."}, "image_hex_truncated":"iVBORw0KGgo...", "screenshot_path":"screenshots/step_0001.png"}
```

Screenshots are saved per-step when `save_screenshots = true`.

### Markdown Report

Generate a human-readable report from any log file:

```bash
tester --report playthrough_log.jsonl
# Creates playthrough_log.md with:
#   - Summary stats (steps, duration, action breakdown)
#   - LLM latency stats (avg/min/max)
#   - Step-by-step decision table
#   - Detailed per-step sections with screenshots
```

### Run Summary

At the end of every run, the harness prints a summary:

```
============================================================
📊 RUN SUMMARY
============================================================
  Steps completed : 42
  Completion      : completed
  Duration        : 127.3s
  Actions         : click=18, press=12, wait=8, done=1, type=3
  Log file        : playthrough_log.jsonl
============================================================
```

---

## 🏗️ Project Structure

```
src/tester/
├── __init__.py     # Package metadata
├── __main__.py     # python -m tester entry
├── cli.py          # CLI argument parsing + subcommands
├── config.py       # Settings (TOML + env vars)
├── client.py       # OpenAI-compatible LLM client with retries
├── harness.py      # Main playthrough loop
├── launcher.py     # Game process launcher
├── models.py       # Pydantic data models
├── prompts.py      # Prompt templates
└── screen.py       # Screen capture

tests/
├── conftest.py     # Shared fixtures
├── test_models.py  # Model validation tests
├── test_config.py  # Config loading tests
├── test_harness.py # Action execution + bounds tests
├── test_launcher.py# Launcher command tests
├── test_prompts.py # Prompt generation tests
└── test_runs.py    # Run namespacing + manifest + git capture tests

deploy/             # Scheduled deployment templates (systemd + Xvfb guide)
```

---

## 🔮 Extending

- **Add a game engine**: Subclass `Launcher` and register it in `launcher._LAUNCHER_MAP`
- **Swap the LLM client**: Implement `LLMClient` (e.g., Anthropic, Gemini, local model)
- **Custom prompts**: Set `system_prompt` / `user_prompt_template` in your TOML config
- **Headless CI**: Install `python-xlib` and use `headless = true` with Xvfb on Linux

---

## 🧑‍💻 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Run all CI checks locally
pytest && ruff check src/ tests/
```

---

## 📦 Dependencies

- `openai>=1.0` — LLM client
- `pydantic>=2.0` + `pydantic-settings` — Config & response models
- `mss` — Fast cross-platform screen capture
- `pillow` — Image handling
- `pyautogui` — Mouse/keyboard automation

---

## 📄 License

MIT — see [LICENSE](LICENSE).