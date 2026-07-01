"""CLI entry point for the tester package.

Usage::

    # Run with a config file
    tester --config my_game.toml

    # Generate a default config file to customise
    tester --new-config > my_game.toml

    # Run using only environment variables (no TOML file)
    TESTER_GAME__TYPE=renpy TESTER_GAME__PATH=./my_game tester
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from .config import Settings
from .client import OpenAIClient
from .launcher import create_launcher, Launcher
from .harness import Harness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tester",
        description="Vision-based autonomous game testing harness.",
        epilog=textwrap.dedent("""\
            Examples:
              tester --config my_game.toml
              TESTER_GAME__PATH=./my_renpy_game tester --config my_game.toml
              tester --new-config > settings.toml
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
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
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
    config_text = textwrap.dedent("""\
        # =============================================================================
        # tester — configuration file
        # =============================================================================
        # Copy this file, rename it, and edit the values below.
        # Environment variables (TESTER_*) take precedence over values in this file.
        # See `tester --help` for details.
        # =============================================================================

        [llm]
        # Any OpenAI-compatible endpoint: OpenAI, Azure, vLLM, Ollama, LiteLLM, etc.
        api_base = "https://api.openai.com/v1"
        api_key = "sk-..."  # Or set TESTER_API_KEY / OPENAI_API_KEY env var
        model = "gpt-4o"
        max_tokens = 600
        temperature = 0.1

        [game]
        type = "renpy"           # "renpy", "godot", or "custom"
        path = "./my_game"       # Path to game directory or executable
        # executable = "renpy.sh"  # Optional: override auto-detected executable name
        resolution = [1280, 720]
        git_pull = false

        [harness]
        max_steps = 50
        wait_after_action = 1.5
        stuck_threshold = 3
        screen_scale = 1.0
        headless = false

        [logging]
        log_file = "playthrough_log.jsonl"
        save_screenshots = true
        screenshot_dir = "screenshots"
        context_window_size = 3
        log_level = "INFO"
    """)
    sys.stdout.write(config_text)


def cmd_list_games(directory: str) -> None:
    """Scan *directory* for recognised game projects and print them."""
    import os
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

    # Load settings (from TOML, env vars, or both)
    try:
        settings = load_settings(args)
    except Exception as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(settings, verbose=args.verbose)
    logger = logging.getLogger("tester.cli")
    logger.info("Configuration loaded | game=%s | endpoint=%s", settings.game.path, settings.llm.api_base)

    # Instantiate LLM client
    api_key = settings.effective_api_key()
    llm_client = OpenAIClient(
        api_base=settings.llm.api_base,
        api_key=api_key,
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
    )

    # Instantiate launcher
    try:
        launcher: Launcher = create_launcher(settings.game)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Failed to create game launcher: %s", exc)
        sys.exit(1)

    # Build and run harness
    harness = Harness(settings, launcher, llm_client)
    harness.run()


if __name__ == "__main__":
    main()
