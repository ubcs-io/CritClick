# Running tester on Windows

This guide covers running `tester` interactively on a Windows development
machine (the game renders to your real desktop). For headless Linux server
deployments, see [`deploy/README.md`](../deploy/README.md).

> Windows support is scaffolded alongside the macOS-first effort. Capture
> and input work, but some edge cases (UAC prompts, Session 0 isolation,
> per-monitor DPI) are still being characterised — please file findings.

## 1. Install

Install **Python 3.10+** from [python.org](https://www.python.org/downloads/)
(tick **"Add Python to PATH"** during install). Then in PowerShell or cmd:

```powershell
git clone <repo> tester
cd tester
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

`pyautogui`, `mss`, and `pillow` install cleanly on Windows with
pre-built wheels.

## 2. Permissions / security notes

Unlike macOS, Windows does not require explicit per-app consent for screen
capture or input synthesis in an interactive session. The main gotchas are:

- **Windows Defender / antivirus** may flag `pyautogui`-driven subprocess
  launches or unsigned game `.exe` files. Add an exclusion for your game
  directory if needed.
- **SmartScreen** ("Windows protected your PC") can block downloaded game
  binaries — click *More info → Run anyway*, or
  `Unblock-File .\game.exe` in PowerShell.
- **UAC elevation**: `tester` should run at the same integrity level as the
  game. A non-elevated `tester` cannot send input to an elevated game
  window.

## 3. Configure for a desktop run

The key setting is `[harness].headless = false`. This keeps the launcher
from passing Godot's `--no-window` and leaves the environment untouched
(`DISPLAY` is irrelevant on Windows).

```powershell
tester --new-config > my_game.toml
```

Minimal Windows-safe config:

```toml
[llm]
api_base = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[game]
type       = "renpy"
path       = "C:\\Games\\my_renpy_game"
executable = "renpy.exe"          # renpy.exe on Windows (NOT renpy.sh)
resolution = [1280, 720]

[harness]
headless          = false         # ← critical on Windows
startup_countdown = true          # gives you time to focus the game window
```

Notes:
- Escape backslashes in TOML (`C:\\Games\\...`) or use forward slashes.
- The Ren'Py launcher auto-selects `renpy.exe` on `win32`; you can also set
  `executable` explicitly.

## 4. Validate, then run

```powershell
# Validate config + executable + API connectivity without launching
tester --check --config my_game.toml

# Confirm screen capture and input are working
tester --self-test --config my_game.toml

# Run the playthrough (the game window will open on your desktop)
tester --config my_game.toml
```

The startup countdown gives you a few seconds to click the game window so
it's focused before input begins. Mouse to any screen corner to abort
(pyautogui failsafe).

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `tester` not recognised | venv not active, or Python not on PATH | `.venv\Scripts\activate`; reinstall Python with PATH option |
| Game opens but no input happens | Integrity-level mismatch (UAC) | Run `tester` and the game at the same elevation |
| `FileNotFoundError: renpy.exe` | Wrong executable name | Set `[game].executable = "renpy.exe"` |
| Game blocked on launch | SmartScreen / Defender | `Unblock-File`; add Defender exclusion |
| Coordinates off on high-DPI | Per-monitor DPI scaling | adjust `[harness].screen_scale` |
| Lock-file check is best-effort | Windows has no `fcntl.flock` | `tester` falls back to a non-empty-file check |

## 6. Scheduling on Windows

For unattended/repeated runs, use **Task Scheduler**. A registration script
and instructions live in [`deploy/windows/`](../deploy/windows/). The task
must be set to **"Run only when user is logged on"** so it runs in an
interactive desktop session where screen capture and input are possible.