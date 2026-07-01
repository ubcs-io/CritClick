"""tester — Vision-based autonomous game testing harness.

Supports any OpenAI-compatible LLM endpoint (OpenAI, Azure, vLLM, Ollama, LiteLLM, etc.)
for vision-driven playthroughs of Ren'Py, Godot, or custom game engines.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # Python < 3.8 fallback (not expected, but defensive)
    from importlib_metadata import PackageNotFoundError, version  # type: ignore[no-redef]

try:
    __version__ = version("tester")
except PackageNotFoundError:  # Not installed (e.g. running from source without install)
    __version__ = "0.0.0+unknown"
