# =============================================================================
# tester — Windows Task Scheduler registration script
# =============================================================================
# Schedules a `tester` playthrough on Windows. Screen capture and input
# synthesis only work in an interactive desktop session, so the task is
# registered with -LogonType Interactive ("Run only when user is logged on").
#
# USAGE
# -----
#   .\deploy\windows\tester-task.ps1 `
#       -GameName  "mygame" `
#       -TesterBin "C:\tester\.venv\Scripts\tester.exe" `
#       -Config    "C:\tester\mygame.toml" `
#       -WorkDir   "C:\tester" `
#       -Hour      2 `
#       -Minute    30
#
# REQUIREMENTS
# ------------
# - Grant any needed Defender exclusions / SmartScreen unblocks for the game
#   binary (see docs/DEV_WINDOWS.md).
# - [harness].headless = false in the config (no Xvfb path on Windows).
# - Run this script in an elevated or normal PowerShell session as the user
#   who will own the task.
# =============================================================================
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string] $GameName,
    [Parameter(Mandatory=$true)] [string] $TesterBin,
    [Parameter(Mandatory=$true)] [string] $Config,
    [Parameter(Mandatory=$true)] [string] $WorkDir,
    [Parameter(Mandatory=$true)] [int]    $Hour,
    [Parameter(Mandatory=$true)] [int]    $Minute,
    [string] $RunsDir = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TesterBin)) { throw "Tester binary not found: $TesterBin" }
if (-not (Test-Path $Config))    { throw "Config file not found: $Config" }
if (-not (Test-Path $WorkDir))   { throw "WorkDir not found: $WorkDir" }

$TaskName = "io.ubcs.tester.$GameName"

# Resolve runs dir: explicit override, else <WorkDir>\runs
if ($RunsDir) {
    $resolvedRunsDir = $RunsDir
} else {
    $resolvedRunsDir = Join-Path $WorkDir "runs"
}

# Build the argument string passed to tester.exe
$runId   = "$(Get-Date -Format 'yyyyMMddTHHmmss')"
$argList = @(
    "--config",   $Config,
    "--run-id",   $runId,
    "--runs-dir", $resolvedRunsDir
) -join " "

$Action = New-ScheduledTaskAction `
    -Execute $TesterBin `
    -Argument $argList `
    -WorkingDirectory $WorkDir

# Daily trigger at the specified local time
$Trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]"$Hour`:$Minute")

# Interactive (logged-on) settings — required for screen capture and input.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "tester autonomous playthrough ($GameName)" `
    -Force

Write-Host "Registered task: $TaskName"
Write-Host "Trigger now:     Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Uninstall:       Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"