# Windows scheduling (Task Scheduler)

This directory schedules `tester` playthroughs on Windows via **Task
Scheduler**.

## Requirements

1. **Interactive desktop session** — screen capture and input synthesis only
   work in a logged-on user's desktop session. The task is registered with
   `-LogonType Interactive` ("Run only when user is logged on"). Do **not**
   use "Run whether user is logged on or not" — that runs in Session 0 and
   cannot capture or send input.
2. **Same integrity level as the game** — if the game runs elevated, the
   task must too (and vice-versa); otherwise input injection silently fails.
3. **`headless = false`** in `[harness]` of your config (no Xvfb on Windows).
4. **Absolute paths** — Task Scheduler does not honour your shell PATH or
   venv activation. Use the absolute path to the venv's `tester.exe`, e.g.
   `C:\tester\.venv\Scripts\tester.exe`.

## Install

```powershell
# From the repo root, as the user who will own the task:
.\deploy\windows\tester-task.ps1 `
    -GameName  "mygame" `
    -TesterBin "C:\tester\.venv\Scripts\tester.exe" `
    -Config    "C:\tester\mygame.toml" `
    -WorkDir   "C:\tester" `
    -Hour      2 `
    -Minute    30
```

If PowerShell blocks the script, allow it once for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run now / inspect / uninstall

```powershell
# Trigger immediately
Start-ScheduledTask -TaskName "io.ubcs.tester.mygame"

# Inspect last result and history
Get-ScheduledTaskInfo -TaskName "io.ubcs.tester.mygame"
Get-WinEvent -LogName "Microsoft-Windows-TaskScheduler/Operational" |
    Where-Object { $_.Message -like "*io.ubcs.tester.mygame*" } |
    Select-Object -First 10 TimeCreated, Id, Message

# Uninstall
Unregister-ScheduledTask -TaskName "io.ubcs.tester.mygame" -Confirm:$false
```

## Logs

`tester` writes stdout/stderr into the runs dir under
`runs\<run_id>\playthrough_log.jsonl` and a `run_manifest.json`. Task
Scheduler also captures the process exit code in
`Get-ScheduledTaskInfo.LastTaskResult`.

## Notes / gotchas

- **Lock file** — for scheduled runs, extend `tester-task.ps1` to add
  `--lock-file` so an overrunning playthrough blocks the next slot. (Windows
  has no `fcntl.flock`; `tester` falls back to a non-empty-file check.)
- **Defender / SmartScreen** — pre-approve the game binary (exclusion or
  `Unblock-File`) so the scheduled run isn't blocked silently.
- **Console window** — the registered action may briefly pop a console
  window. If that's disruptive, wrap `tester.exe` in a `wscript.exe`-based
  hidden launcher (out of scope for this template).