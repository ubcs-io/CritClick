# Running tester on macOS

This guide covers running `tester` interactively on a macOS development
machine (the game renders to your real desktop instead of an Xvfb virtual
display). For headless Linux server deployments, see
[`deploy/README.md`](../deploy/README.md).

## 1. Install

```bash
git clone <repo> tester && cd tester
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+ is required. `pyautogui`, `mss`, and `pillow` are all
cross-platform and install cleanly on macOS.

## 2. Grant macOS permissions (do this first — failures here are silent)

`tester` both **captures the screen** and **synthesises mouse/keyboard
input**, so macOS requires two TCC (Transparency, Consent, and Control)
authorisations. Without them, captures come back blank and clicks do
nothing — with no error message.

1. **Screen Recording** (required by `mss`)
   - Open *System Settings → Privacy & Security → Screen Recording*.
   - Add the app you launch `tester` from:
     - **Terminal.app** / **iTerm.app** — add the terminal.
     - **VS Code** — add `Visual Studio Code` (or run from the integrated
       terminal after approving it).
     - **A virtualenv** — approve the parent app that owns the process.
   - Toggle the switch on, then **quit and relaunch** that app. macOS only
     re-checks permission at launch.

2. **Accessibility** (required by `pyautogui` for mouse/keyboard)
   - *System Settings → Privacy & Security → Accessibility*.
   - Add the same app(s) as above and enable them.

> After granting, run `tester --self-test` (see below) to confirm both
> pathways are working before starting a real playthrough.

## 3. Configure for a desktop run

The only setting that matters for macOS is `[harness].headless = false`.
This tells the launcher **not** to touch `DISPLAY` and **not** to pass
Godot's `--no-window`, so the game window appears on your desktop.

Copy the example config and edit the game path:

```bash
tester --new-config > my_game.toml
```

Minimal macOS-safe config:

```toml
[llm]
api_base = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[game]
type       = "renpy"
path       = "/Users/you/Games/my_renpy_game"
executable = "renpy.sh"          # renpy.sh on macOS (NOT renpy.exe)
resolution = [1280, 720]

[harness]
headless          = false        # ← critical on macOS
startup_countdown = true         # gives you time to focus the game window
```

## 4. Gatekeeper / quarantine notes

Downloaded games are often quarantined by Gatekeeper:

- **`.app` bundles** — right-click → *Open* once to approve, or
  `xattr -dr com.apple.quarantine /path/to/Game.app`.
- **`renpy.sh`** — the launcher script must be executable:
  ```bash
  chmod +x /Users/you/Games/my_renpy_game/renpy.sh
  ```
  If the engine binary inside the `.app` is quarantined, clear it the same
  way with `xattr -dr`.

## 5. Validate, then run

```bash
# Validate config + executable + API connectivity without launching
tester --check --config my_game.toml

# Confirm screen capture and input permissions are working
tester --self-test --config my_game.toml

# Run the playthrough (the game window will open on your desktop)
tester --config my_game.toml
```

The startup countdown gives you a few seconds to click the game window so
it's focused before input begins. Mouse to any screen corner to abort
(pyautogui failsafe).

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Screenshots are blank / black | Screen Recording not granted | Grant in Privacy & Security, relaunch terminal |
| No clicks / typing happen | Accessibility not granted | Grant in Privacy & Security, relaunch terminal |
| `Permission denied: renpy.sh` | Script not executable | `chmod +x renpy.sh` |
| Game won't open / "damaged" | Gatekeeper quarantine | `xattr -dr com.apple.quarantine <path>` |
| Coordinates seem off on Retina | DPI scaling | adjust `[harness].screen_scale` |
| `tester` not found on PATH | venv not active | `source .venv/bin/activate` |

## 7. Scheduling on macOS

For unattended/repeated runs, use **launchd** (macOS's native scheduler).
A template and instructions live in
[`deploy/macos/`](../deploy/macos/). Note that launchd jobs that need
screen capture / input must run in a **GUI login session** — they cannot
capture the screen from a pure background context.