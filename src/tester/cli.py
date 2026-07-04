"""CLI entry point for the tester package.

Usage::

    # Run with a config file
    tester --config my_game.toml

    # Generate a default config file to customise
    tester --new-config > my_game.toml

    # Validate config, executable paths, and API connectivity without launching
    tester --check --config my_game.toml

    # Run the analysis loop without executing actions or launching the game
    tester --dry-run --config my_game.toml

    # Convert a JSONL log into a readable Markdown report
    tester --report playthrough_log.jsonl

    # Run using only environment variables (no TOML file)
    TESTER_GAME__TYPE=renpy TESTER_GAME__PATH=./my_game tester
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from .client import OpenAIClient
from .config import Settings
from .harness import Harness
from .launcher import Launcher, create_launcher

# ---------------------------------------------------------------------------
# Screen size presets for --mobile / --tablet flags
# ---------------------------------------------------------------------------

SCREEN_PRESETS: dict[str, tuple[int, int]] = {
    "mobile": (390, 844),   # iPhone 14 Pro (portrait)
    "tablet": (1024, 768),  # iPad (portrait)
}



def apply_screen_preset(args: argparse.Namespace, game_config: "GameConfig") -> None:
    """Apply mobile/tablet screen size overrides to ``game_config.resolution``.

    Raises ``SystemExit`` via ``parser.error()`` if conflicting options are used,
    so the caller must pass the argument parser instance.
    """
    # Validate mutual exclusivity (mobile vs tablet) and landscape dependency.
    # We can't use the parser's error() here easily, so we just print to stderr
    # and exit. The parser argument is accepted for future compatibility.
    if args.mobile and args.tablet:
        print(
            "Error: --mobile and --tablet are mutually exclusive. Use one or neither.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.mobile and not args.tablet:
        if args.landscape:
            print(
                "Warning: --landscape has no effect without --mobile or --tablet. Ignoring.",
                file=sys.stderr,
            )
        return

    preset_key = "mobile" if args.mobile else "tablet"
    w, h = SCREEN_PRESETS[preset_key]

    if args.landscape:
        w, h = h, w  # swap width/height for landscape orientation

    game_config.resolution = (w, h)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tester",
        description="Vision-based autonomous game testing harness.",
        epilog=textwrap.dedent("""\
            Examples:
              tester --config my_game.toml
              TESTER_GAME__PATH=./my_renpy_game tester --config my_game.toml
              tester --new-config > settings.toml
              tester --check --config my_game.toml
              tester --dry-run --config my_game.toml
              tester --report playthrough_log.jsonl
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to a TOML configuration file.",
    )
    parser.add_argument(
        "--new-config",
        action="store_true",
        help="Print a default configuration file to stdout and exit.",
    )
    parser.add_argument(
        "--list-games",
        type=str,
        default=None,
        metavar="DIR",
        help="Scan a directory for recognised game projects (Ren'Py / Godot) and exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config, executable paths, and API connectivity, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the analysis loop without launching the game or executing actions.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Verify screen capture and input permissions work on this machine, "
            "then exit. Useful on macOS/Windows dev machines to catch missing "
            "TCC / accessibility permissions before a real playthrough."
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="LOGFILE",
        help="Convert a JSONL playthrough log into a Markdown report and exit.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="Log full LLM prompts and raw responses for debugging.",
    )
    parser.add_argument(
        "--debug-harness",
        action="store_true",
        help="Log per-step action parsing, coordinate scaling, bounds checks, and stuck detection details.",
    )
    parser.add_argument(
        "--debug-screen",
        action="store_true",
        help="Log screen capture details, coordinate transformations, accessibility checks, and click position tracking.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Namespace this run's outputs under <runs-dir>/<run-id>/. "
            "When set, a run_manifest.json is written for ingestion. "
            "Omit for the legacy flat output layout."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Root directory for namespaced run outputs (default: runs).",
    )
    parser.add_argument(
        "--lock-file",
        type=str,
        default=None,
        help=(
            "Path to a lock file. If the file is already held by another run, "
            "this invocation exits with code 4 (EXIT_LOCKED) instead of starting. "
            "Recommended for scheduled deployments on a shared display."
        ),
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        help=(
            "Override game resolution to mobile portrait (390×844, iPhone 14 Pro). "
            "Use --landscape to swap to 844×390. Mutually exclusive with --tablet."
        ),
    )
    parser.add_argument(
        "--tablet",
        action="store_true",
        help=(
            "Override game resolution to tablet portrait (1024×768, iPad). "
            "Use --landscape to swap to 768×1024. Mutually exclusive with --mobile."
        ),
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help=(
            "Swap the --mobile or --tablet preset to landscape orientation "
            "(width ↔ height). Has no effect without --mobile or --tablet."
        ),
    )

    return parser


def load_settings(args: argparse.Namespace) -> Settings:
    """Load settings from config file, env vars, or both."""
    if args.config:
        return Settings.from_toml(args.config)
    return Settings.from_env_only()


def setup_logging(settings: Settings, verbose: bool = False) -> None:
    """Configure the root tester logger."""
    level = logging.DEBUG if verbose else getattr(logging, settings.logging.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_new_config() -> None:
    """Print a default configuration file to stdout."""
    # Read from the bundled example to avoid maintaining a second copy.
    import os
    from pathlib import Path

    candidates = [
        Path(__file__).parent.parent.parent / "settings.example.toml",
        Path(os.getcwd()) / "settings.example.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            sys.stdout.write(candidate.read_text(encoding="utf-8"))
            return

    # Fallback to inline minimal config if example file not found
    sys.stdout.write(textwrap.dedent("""\
        [llm]
        api_base = "https://api.openai.com/v1"
        model = "gpt-4o"

        [game]
        type = "renpy"
        path = "./my_game"
    """))


def cmd_list_games(directory: str) -> None:
    """Scan *directory* for recognised game projects and print them."""
    from pathlib import Path

    base = Path(directory)
    if not base.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    found = False

    # Ren'Py detection: look for renpy.sh or renpy.exe
    for renpy_name in ("renpy.sh", "renpy.exe"):
        p = base / renpy_name
        if p.is_file():
            print(f"renpy    {p}")
            found = True

    # Ren'Py alternate location
    for renpy_name in ("renpy.sh", "renpy.exe"):
        p = base / "renpy" / renpy_name
        if p.is_file():
            print(f"renpy    {p}")
            found = True

    # Godot detection: project.godot files
    for p in base.rglob("project.godot"):
        print(f"godot    {p}")
        found = True

    if not found:
        print(f"No recognised game projects found in '{directory}'.")
        print("Looked for: renpy.sh/renpy.exe, project.godot")


def cmd_check(settings: Settings) -> int:
    """Validate config, executable paths, and API connectivity. Returns exit code."""
    ok = True

    # 1. Config is loadable (already passed if we got here)
    print("✅ Configuration loaded successfully.")
    print(f"   Game type : {settings.game.type}")
    print(f"   Game path : {settings.game.path}")
    print(f"   Model     : {settings.llm.model}")
    print(f"   Endpoint  : {settings.llm.api_base}")

    api_key = settings.effective_api_key()
    if api_key:
        print(f"   API key   : {'set (' + str(len(api_key)) + ' chars)'}")
    else:
        print("   API key   : NOT SET (may work for local endpoints)")

    # 2. Executable resolves
    print("\n🔎 Checking game executable…")
    try:
        launcher = create_launcher(settings.game, headless=settings.harness.headless)
        exe = launcher.resolve_executable()
        print(f"✅ Executable found: {exe}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ Executable error: {exc}")
        ok = False

    # 3. API connectivity (lightweight models.list call)
    print("\n🔎 Checking API connectivity…")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or "", base_url=settings.llm.api_base.rstrip("/"))
        client.models.list()
        print("✅ API endpoint reachable.")
    except Exception as exc:
        print(f"⚠️  API connectivity check failed: {exc}")
        print("   (This may be normal for some local endpoints that don't implement /models.)")

    print()
    if ok:
        print("✅ Pre-flight check passed — ready to run.")
        return 0
    print("❌ Pre-flight check found issues. Fix the above before running.")
    return 1


def cmd_report(logfile: str) -> int:
    """Convert a JSONL playthrough log into a Markdown report. Returns exit code."""
    import json
    from pathlib import Path

    p = Path(logfile)
    if not p.is_file():
        print(f"Log file not found: {logfile}", file=sys.stderr)
        return 1

    entries = []
    with open(p, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"⚠️  Skipping invalid JSON on line {line_num}: {exc}", file=sys.stderr)

    if not entries:
        print("Log file contains no valid entries.", file=sys.stderr)
        return 1

    # Build Markdown
    out = []
    out.append("# Playthrough Report\n")
    out.append(f"**Log file:** `{logfile}`\n")
    out.append(f"**Total steps:** {len(entries)}\n")

    # Timing stats
    durations = [e.get("duration_ms", 0) for e in entries if e.get("duration_ms")]
    if durations:
        avg_ms = sum(durations) / len(durations)
        max_ms = max(durations)
        min_ms = min(durations)
        out.append(
            f"**LLM latency:** avg {avg_ms:.0f}ms, min {min_ms}ms, max {max_ms}ms\n"
        )

    # Action counts
    action_counts: dict[str, int] = {}
    for e in entries:
        action = e.get("llm_response", {}).get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    if action_counts:
        out.append("\n## Action Breakdown\n")
        out.append("| Action | Count |")
        out.append("|--------|-------|")
        for action, count in sorted(action_counts.items()):
            out.append(f"| {action} | {count} |")
        out.append("")

    # Step-by-step table
    out.append("\n## Step-by-Step Decisions\n")
    out.append("| Step | Action | Description | Duration |")
    out.append("|------|--------|-------------|----------|")
    for e in entries:
        step = e.get("step", "?")
        resp = e.get("llm_response", {})
        action = resp.get("action", "?")
        desc = resp.get("description", "").replace("|", "\\|")
        narrative = resp.get("narrative", "")
        dur = e.get("duration_ms", 0)
        out.append(f"| {step} | {action} | {desc} ({narrative}) | {dur}ms |")
    out.append("")

    # Detailed step list with screenshots
    out.append("\n## Detailed Steps\n")
    for e in entries:
        step = e.get("step", "?")
        resp = e.get("llm_response", {})
        narrative = resp.get("narrative", "")
        reasoning = resp.get("reasoning", "")
        screenshot = e.get("screenshot_path")

        out.append(f"### Step {step}: {narrative}\n")
        out.append(f"- **Description:** {resp.get('description', '')}")
        out.append(f"- **Action:** `{resp.get('action', '')}`")
        if resp.get("coordinates"):
            out.append(f"- **Coordinates:** {resp['coordinates']}")
        if resp.get("text_to_type"):
            out.append(f"- **Text typed:** `{resp['text_to_type']}`")
        if resp.get("key_to_press"):
            out.append(f"- **Key pressed:** `{resp['key_to_press']}`")
        out.append(f"- **Reasoning:** {reasoning}")
        if screenshot:
            out.append(f"\n![Step {step}]({screenshot})")
        out.append("")

    report = "\n".join(out)
    report_path = p.with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(f"✅ Report written to: {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def cmd_self_test(settings: Settings | None) -> int:
    """Verify screen capture and input synthesis work on this machine.

    Returns 0 if both pathways are functional, 1 otherwise. On macOS this
    catches missing Screen Recording / Accessibility TCC grants; on Windows
    it catches integrity-level mismatches indirectly (input silently no-ops).
    """
    import platform
    import sys as _sys

    print(f"tester self-test on {platform.platform()}")
    print()

    scale = settings.harness.screen_scale if settings else 1.0
    ok = True

    # 1. Screen capture
    print("🔎 Testing screen capture (mss / pyautogui)…")
    try:
        from .screen import Capturer

        capturer = Capturer(scale=scale)
        img = capturer.capture_pil()
        w, h = img.size
        # Heuristic: an all-black capture often indicates a permission issue
        # (especially on macOS where TCC denial returns a black frame, not an
        # exception). Sample a few pixels; if every sampled pixel is identical,
        # warn.
        samples = [img.getpixel((w // 4, h // 4)), img.getpixel((w // 2, h // 2)),
                   img.getpixel((3 * w // 4, 3 * h // 4))]
        all_same = len(set(samples)) == 1

        print(f"✅ Capture OK — {w}x{h} via {('mss' if capturer._mss_available else 'pyautogui')}.")
        if all_same and all(s == samples[0] for s in samples):
            # Single-color screen is suspicious on a real desktop
            print(f"⚠️  Captured frame is a uniform color ({samples[0]}).")
            if _sys.platform == "darwin":
                print("   On macOS this usually means Screen Recording permission")
                print("   was not granted to the terminal/IDE. See docs/DEV_MACOS.md.")
            else:
                print("   This may indicate a headless/display issue or a blank screen.")
        print(f"   Saved sample to: tester_selftest_capture.png")
        img.save("tester_selftest_capture.png")
    except Exception as exc:
        print(f"❌ Capture failed: {exc}")
        if _sys.platform == "darwin":
            print("   Grant Screen Recording permission and relaunch the terminal/IDE.")
        ok = False

    # 2. Input synthesis
    print("\n🔎 Testing input synthesis (pyautogui)…")
    try:
        import pyautogui

        # size() is a safe, non-destructive call that exercises the backend
        size = pyautogui.size()
        print(f"✅ pyautogui OK — screen reported as {size.width}x{size.height}.")
        if _sys.platform == "darwin":
            print("   (Accessibility permission grants mouse/keyboard control.)")
    except Exception as exc:
        print(f"❌ pyautogui failed: {exc}")
        if _sys.platform == "darwin":
            print("   Grant Accessibility permission and relaunch the terminal/IDE.")
        ok = False

    # 3. Platform-specific guidance
    print()
    if _sys.platform == "darwin":
        print("ℹ️  macOS: System Settings → Privacy & Security →")
        print("     • Screen Recording  — add the app running tester")
        print("     • Accessibility     — add the app running tester")
        print("   Then quit and relaunch that app. See docs/DEV_MACOS.md.")
    elif _sys.platform == "win32":
        print("ℹ️  Windows: if input silently fails, check that tester and the")
        print("   game run at the same UAC integrity level. See docs/DEV_WINDOWS.md.")

    print()
    if ok:
        print("✅ Self-test passed — capture and input pathways are working.")
        return 0
    print("❌ Self-test found issues. Fix the above before running a playthrough.")
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle special commands that don't need settings
    if args.version:
        from . import __version__
        print(f"tester v{__version__}")
        return

    if args.new_config:
        cmd_new_config()
        return

    if args.list_games:
        cmd_list_games(args.list_games)
        return

    # --report needs a logfile but no full settings
    if args.report:
        exit_code = cmd_report(args.report)
        sys.exit(exit_code)

    # --self-test: verify capture + input permissions, then exit.
    # Doesn't strictly need a config, but uses [harness].screen_scale if present.
    if args.self_test:
        try:
            settings = load_settings(args)
        except Exception:
            settings = None  # self-test can run without a config
        sys.exit(cmd_self_test(settings))

    # Load settings (from TOML, env vars, or both)
    try:
        settings = load_settings(args)
    except Exception as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(settings, verbose=args.verbose)
    logger = logging.getLogger("tester.cli")

    # Apply screen size overrides from CLI flags before anything reads resolution
    apply_screen_preset(args, settings.game)
    if args.mobile or args.tablet:
        w, h = settings.game.resolution
        logger.info("🖥️  Screen override applied: %dx%d", w, h)

    logger.info("Configuration loaded | game=%s | endpoint=%s", settings.game.path, settings.llm.api_base)

    # --check: validate and exit
    if args.check:
        exit_code = cmd_check(settings)
        sys.exit(exit_code)

    # Instantiate LLM client
    api_key = settings.effective_api_key()
    llm_client = OpenAIClient(
        api_base=settings.llm.api_base,
        api_key=api_key,
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        max_retries=settings.llm.max_retries,
        retry_delay=settings.llm.retry_delay,
        debug_llm=args.debug_llm,
    )

    # Instantiate launcher (skip in dry-run)
    launcher: Launcher
    if args.dry_run:
        from .launcher import Launcher as _Launcher
        # Create a minimal stub launcher for dry-run mode
        class _StubLauncher(_Launcher):
            def resolve_executable(self) -> str:
                return ""
            def build_command(self, executable: str) -> list[str]:
                return []
        launcher = _StubLauncher(settings.game)
    else:
        try:
            launcher = create_launcher(settings.game, headless=settings.harness.headless)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Failed to create game launcher: %s", exc)
            sys.exit(1)

    # Acquire run lock (prevents overlapping runs on a shared display).
    # Uses fcntl.flock on Linux/macOS; falls back to Windows-style file check.
    lock_handle = None
    if args.lock_file:
        from pathlib import Path

        Path(args.lock_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_handle = open(args.lock_file, "w")
        except OSError as exc:
            logger.error("Could not open lock file %s: %s", args.lock_file, exc)
            sys.exit(1)

        try:
            import fcntl

            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            # Windows: no fcntl. Best-effort: check if file non-empty.
            if lock_handle.read():
                logger.warning("⚠️  Lock file %s appears held; aborting.", args.lock_file)
                lock_handle.close()
                from .harness import EXIT_LOCKED
                sys.exit(EXIT_LOCKED)
        except OSError:
            logger.warning("⚠️  Another run holds the lock (%s); exiting.", args.lock_file)
            lock_handle.close()
            from .harness import EXIT_LOCKED
            sys.exit(EXIT_LOCKED)

        # Write our PID for diagnostics
        try:
            import os as _os

            lock_handle.write(str(_os.getpid()))
            lock_handle.flush()
        except OSError:
            pass

    try:
        # Build and run harness
        harness = Harness(
            settings,
            launcher,
            llm_client,
            dry_run=args.dry_run,
            run_id=args.run_id,
            runs_dir=args.runs_dir,
            debug_harness=args.debug_harness,
            debug_screen=args.debug_screen,
        )
        exit_code = harness.run()
    finally:
        # Release the lock
        if lock_handle is not None:
            try:
                import fcntl

                fcntl.flock(lock_handle, fcntl.LOCK_UN)
            except ImportError:
                pass
            try:
                lock_handle.close()
                import os as _os

                if args.lock_file and _os.path.exists(args.lock_file):
                    _os.remove(args.lock_file)
            except OSError:
                pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
