# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-01

### Added
- **`press` action** — New action type for pressing single keys (Enter, Escape, Space, etc.) via `key_to_press` field. Critical for games that use keyboard shortcuts.
- **LLM retry with exponential backoff** — Transient LLM failures (rate limits, timeouts, invalid JSON) are now retried up to `max_retries` times. Configurable via `[llm].max_retries` and `[llm].retry_delay`.
- **Pre-flight validation (`--check`)** — New `tester --check` command validates config, executable paths, and API connectivity without launching the game.
- **Dry-run mode (`--dry-run`)** — New `tester --dry-run` flag runs the full capture → analyse → log loop without launching the game or executing any actions. Ideal for validating prompts and setup.
- **Markdown report generator (`--report`)** — New `tester --report <logfile>` command converts a JSONL playthrough log into a readable Markdown report with action breakdown, timing stats, and embedded screenshots.
- **Run summary** — On completion, the harness logs a summary with steps completed, completion reason, duration, action breakdown, and log file path.
- **Per-step timing** — `duration_ms` field added to `LogEntry` to track LLM latency per step.
- **Coordinate bounds validation** — Click coordinates outside the expected game window bounds are rejected, preventing accidental clicks on other applications.
- **Startup countdown** — A 3-second countdown with failsafe reminder prints before launching the game (configurable via `[harness].startup_countdown`).
- **Configurable wait duration** — The hardcoded 2s wait is now configurable via `[harness].wait_duration`.
- **Exit codes** — Proper exit codes: `0` (success), `1` (config error), `2` (runtime error), `3` (game died), `130` (interrupted).
- **Unit tests** — New `tests/` directory with coverage for config loading, model parsing, action execution, launcher commands, and prompt generation.
- **Ruff linting** — Added ruff configuration to `pyproject.toml`.
- **GitHub Actions CI** — Workflow runs tests and linting across Python 3.10/3.11/3.12.
- **LICENSE file** — Added MIT license file.
- **`.gitignore`** — Added comprehensive Python/IDE/artifact ignores.

### Changed
- **Version deduplication** — `__version__` now read from package metadata via `importlib.metadata` instead of being hardcoded.
- **`__main__.py` guard** — Added `if __name__ == "__main__":` guard.
- **Config deduplication** — `--new-config` now reads from `settings.example.toml` instead of maintaining a second inline copy.
- **Development status** — Bumped from Alpha to Beta.

### [0.1.0] - Initial Release

- Vision-based capture → analyse → act → log loop
- OpenAI-compatible LLM client abstraction
- Ren'Py, Godot, and custom engine support
- TOML + environment variable configuration
- JSONL logging with optional screenshots
- Stuck detection and context window management