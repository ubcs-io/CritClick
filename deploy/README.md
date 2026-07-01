# Scheduled deployment guide

This guide covers running `tester` as an autonomous, scheduled service on a
headless Linux server. The package itself stays one-shot — an external
scheduler (systemd timers here) triggers a fresh playthrough every few minutes,
and each run writes a self-contained manifest for your agentic ingestion
system to read.

---

## 1. Layout

```
/etc/tester/<game>.toml          # one config per game under test
/var/lib/tester/runs/            # all run output lives here
  └── 20260701T165500Z/
      ├── run_manifest.json      # ← canonical entry point for ingestion
      ├── playthrough_log.jsonl
      └── screenshots/
/run/tester/<game>.lock          # prevents overlapping runs on the display
/usr/local/bin/tester            # installed CLI
```

## 2. Install the package

```bash
sudo useradd --system --create-home --home-dir /var/lib/tester tester
sudo mkdir -p /etc/tester /var/lib/tester/runs /run/tester
sudo chown -R tester:tester /var/lib/tester /run/tester

pip install -e .                 # or: pip install tester
sudo cp deploy/tester@.service deploy/tester@.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Confirm the binary path matches `ExecStart` (`/usr/local/bin/tester` by
default). Adjust the unit file if yours lives elsewhere (`which tester`).

## 3. Headless display (Xvfb)

`tester` drives a real GUI via `mss` + `pyautogui`, so a headless server
needs a virtual framebuffer. Create `/etc/systemd/system/Xvfb.service`:

```ini
[Unit]
Description=X Virtual Framebuffer
After=network.target

[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x720x24
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl enable --now Xvfb.service
```

`pyautogui` on Linux also needs `python-xlib`:

```bash
pip install python-xlib
```

## 4. Add a game config

```bash
sudo cp settings.example.toml /etc/tester/my_game.toml
sudo $EDITOR /etc/tester/my_game.toml   # set path, model, api key, etc.
```

The game `path` can point anywhere on disk — `tester` launches it as a
subprocess and never imports game code.

## 5. Enable the schedule

```bash
sudo systemctl enable --now tester@my_game.timer
systemctl list-timers tester@my_game.timer
```

A new playthrough starts roughly every 5 minutes (with up to 1 minute of
jitter). Each run gets a UTC-timestamp `run_id` and its own output directory.

## 6. Reading results (for the agentic ingestion system)

Discover completed runs by globbing manifests:

```bash
ls /var/lib/tester/runs/*/run_manifest.json
```

Each `run_manifest.json` is a single JSON object:

```json
{
  "run_id": "20260701T165500Z",
  "started_at": "2026-07-01T16:55:00.123456",
  "ended_at": "2026-07-01T17:02:15.987654",
  "duration_seconds": 435.86,
  "exit_code": 0,
  "completion_reason": "completed",
  "game": {"type": "renpy", "path": "/srv/games/my_game", "resolution": [1280, 720]},
  "git": {"commit": "abc1234", "branch": "main", "dirty": false},
  "llm": {"model": "gpt-4o", "api_base": "https://api.openai.com/v1"},
  "steps_completed": 42,
  "action_counts": {"click": 18, "press": 12, "wait": 8, "done": 1, "type": 3},
  "log_file": "/var/lib/tester/runs/20260701T165500Z/playthrough_log.jsonl",
  "screenshot_dir": "/var/lib/tester/runs/20260701T165500Z/screenshots",
  "tester_version": "0.1.0"
}
```

The `git.commit` field is the key for correlating playthrough quality against
code changes over time. The full per-step trace is in the sibling
`playthrough_log.jsonl`; `tester --report <log>` renders it to Markdown.

## 7. Exit codes

| Code | Meaning | Notes |
|------|---------|-------|
| 0 | success | natural completion or `done` action |
| 1 | config error | bad config / missing executable |
| 2 | runtime error | LLM failures, crashes |
| 3 | game died | game process exited unexpectedly — still a valid run |
| 4 | locked | another run holds the lock; this invocation skipped |
| 130 | interrupted | Ctrl+C / SIGINT |

`SuccessExitStatus=0 3 4` in the service unit treats 3 and 4 as clean so
systemd doesn't flag them as failures.

## 8. Running multiple games

Each game gets its own timer + service instance, all sharing the same display:

```bash
sudo systemctl enable --now tester@other_game.timer
```

Lock files are per-instance (`/run/tester/<instance>.lock`), so games don't
block each other — but they *will* share the single Xvfb display, which
means their windows may overlap. For truly parallel runs, run separate
Xvfb instances on different `DISPLAY` numbers and override the environment
per service instance.

## 9. Ad-hoc / one-off runs

The new flags work outside systemd too:

```bash
tester --config my_game.toml \
       --run-id "$(date -u +%Y%m%dT%H%M%SZ)" \
       --runs-dir ./runs \
       --lock-file /tmp/tester.lock
```

Omit `--run-id` to get the original flat-layout behavior
(`playthrough_log.jsonl` in the cwd, no manifest).