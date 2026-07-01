# macOS scheduling (launchd)

This directory schedules `tester` playthroughs on macOS via **launchd**.

## Requirements

1. **GUI login session** — launchd agents that need screen capture / input
   must run in the interactive Aqua session. They **cannot** capture the
   screen from a pure background context, so do not run these as a system
   daemon. Log in and keep the session active (lock screen is fine; logout
   is not).
2. **TCC permissions** — grant **Screen Recording** and **Accessibility** to
   the app that owns the `tester` process (Terminal.app, iTerm, or the GUI
   app). See [`docs/DEV_MACOS.md`](../../docs/DEV_MACOS.md).
3. **`headless = false`** in `[harness]` of your config (no Xvfb on macOS).
4. **Absolute paths** — launchd does not honour your shell PATH or activate
   scripts. Use the absolute path to the venv's `tester` binary, e.g.
   `/Users/you/tester/.venv/bin/tester`.

## Install

```bash
# 1. Customise the template (placeholders are wrapped in {{...}})
cp deploy/macos/io.ubcs.tester.plist.template \
   ~/Library/LaunchAgents/io.ubcs.tester.mygame.plist
$EDITOR ~/Library/LaunchAgents/io.ubcs.tester.mygame.plist

# 2. Load the agent
launchctl load ~/Library/LaunchAgents/io.ubcs.tester.mygame.plist

# 3. (Optional) Trigger immediately instead of waiting for the schedule
launchctl start io.ubcs.tester.mygame

# 4. Tail logs
tail -f /tmp/tester.mygame.out.log
tail -f /tmp/tester.mygame.err.log
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/io.ubcs.tester.mygame.plist
rm ~/Library/LaunchAgents/io.ubcs.tester.mygame.plist
```

## Notes / gotchas

- **Lock file** — for scheduled runs, add `--lock-file` to
  `ProgramArguments` so an overrunning playthrough blocks the next slot.
- **`--run-id` with a date stamp** — the template passes a `date`-based
  `--run-id`; note launchd does **not** expand `$(...)` shell substitutions,
  so either hardcode the id or wrap the invocation in a small shell script
  that the plist calls instead of the `tester` binary directly.
- **Wakeup from sleep** — `StartCalendarInterval` will fire even if the
  machine was asleep at the scheduled time (launchd runs it on wake). If you
  need strict timing, disable power nap / sleep on the host.