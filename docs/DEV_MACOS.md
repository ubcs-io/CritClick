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

## 6. Window focus & coordinate handling

After launching the game, `tester` automatically:

1. **Detects the game window** — it polls for a window belonging to the game
   process PID (up to `[harness].window_find_timeout` seconds, default 15s).
2. **Crops screenshots** to the game window only — the LLM sees just the
   game, not your desktop/menubar/dock.
3. **Offsets click coordinates** — the LLM returns coordinates relative to
   the game window's top-left corner, and `tester` adds the window's screen
   position before clicking.
4. **Re-focuses the window** before every click — using AppleScript to bring
   the game frontmost so input lands correctly.

This means you no longer need to manually click the game window during the
startup countdown — `tester` handles focus automatically.

If no game window is found (e.g. on a headless Xvfb server), the harness
falls back gracefully to full-screen capture and absolute screen coordinates.

### Configuration options

```toml
[harness]
window_find_timeout = 15.0    # max seconds to wait for the game window (0–120)
stuck_recovery      = true     # auto-recover from stuck states (default: true)
```

## 7. Stuck-state recovery

When the harness detects 3 or more identical actions in a row (a "stuck"
state), it can automatically attempt recovery. Recovery strategies rotate
through:

1. Press **Escape** (common "back/menu" key)
2. Press **Space** (common "advance" key)
3. Click the **center** of the game window
4. Extended **wait** (5 seconds)
5. Press **Enter**

This is controlled by `[harness].stuck_recovery` (default: `true`). Disable
it if you prefer the harness to simply warn and continue.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Screenshots are blank / black | Screen Recording not granted | Grant in Privacy & Security, relaunch terminal |
| No clicks / typing happen | Accessibility not granted | Grant in Privacy & Security, relaunch terminal |
| Clicks land on wrong window | Game window not focused | Ensure window_find_timeout is high enough; check logs for "Game window found" |
| `Permission denied: renpy.sh` | Script not executable | `chmod +x renpy.sh` |
| Game won't open / "damaged" | Gatekeeper quarantine | `xattr -dr com.apple.quarantine <path>` |
| Coordinates seem off on Retina | DPI scaling | adjust `[harness].screen_scale` |
| Harness stuck in click loop | LLM confused or game hung | Recovery auto-kicks in; if not, set `stuck_recovery = true` |
| `tester` not found on PATH | venv not active | `source .venv/bin/activate` |

## 9. Scheduling on macOS

For unattended/repeated runs, use **launchd** (macOS's native scheduler).
A template and instructions live in
[`deploy/macos/`](../deploy/macos/). Note that launchd jobs that need
screen capture / input must run in a **GUI login session** — they cannot
capture the screen from a pure background context.