"""Tests for run namespacing, manifest output, git capture, and locking."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from tester.client import LLMClient
from tester.harness import EXIT_LOCKED, EXIT_SUCCESS, Harness
from tester.launcher import capture_git_info

# ---------------------------------------------------------------------------
# Namespacing
# ---------------------------------------------------------------------------

class TestRunNamespace:
    def test_namespace_rewrites_paths(self, sample_settings, tmp_path):
        run_id = "20260701T165500Z"
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        Harness(
            sample_settings,
            launcher,
            client,
            dry_run=True,
            run_id=run_id,
            runs_dir=tmp_path,
        )

        assert sample_settings.logging.log_file == str(tmp_path / run_id / "playthrough_log.jsonl")
        assert sample_settings.logging.screenshot_dir == str(tmp_path / run_id / "screenshots")

    def test_no_explicit_run_id_still_namespaces(self, sample_settings, tmp_path):
        """When no run_id is provided, a timestamp-based default is auto-generated
        and outputs are always namespaced under runs/<auto-id>/."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(
            sample_settings, launcher, client,
            dry_run=True,
            runs_dir=tmp_path,
        )
        assert harness.run_id is not None
        assert len(harness.run_id) > 10  # e.g. "20260704T181316-123456"
        # Paths should be namespaced under runs/<auto-id>/
        assert str(tmp_path / harness.run_id) in sample_settings.logging.log_file

    def test_namespace_creates_run_directory(self, sample_settings, tmp_path):
        run_id = "testrun"
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        Harness(
            sample_settings,
            launcher,
            client,
            dry_run=True,
            run_id=run_id,
            runs_dir=tmp_path,
        )
        # The log file's parent dir is created unconditionally during _setup_output.
        # (sample_settings has save_screenshots=False, so the screenshots dir
        # isn't created here — that path is covered by the integration test.)
        assert (tmp_path / run_id).is_dir()
        assert (tmp_path / run_id / "playthrough_log.jsonl") == Path(
            sample_settings.logging.log_file
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestRunManifest:
    def test_manifest_always_written_with_auto_id(self, sample_settings, tmp_path):
        """Since run_id is always auto-generated, a manifest is always written."""
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(
            sample_settings, launcher, client,
            dry_run=True,
            runs_dir=tmp_path,
        )
        harness._exit_code = EXIT_SUCCESS
        harness.completion_reason = "completed"
        harness._write_manifest()
        manifest_path = tmp_path / harness.run_id / "run_manifest.json"
        assert manifest_path.is_file()

    def test_manifest_contains_required_fields(self, sample_settings, tmp_path):
        run_id = "manifesttest"
        launcher = MagicMock()
        client = MagicMock(spec=LLMClient)
        harness = Harness(
            sample_settings,
            launcher,
            client,
            dry_run=True,
            run_id=run_id,
            runs_dir=tmp_path,
        )
        # Simulate a completed run
        harness._exit_code = EXIT_SUCCESS
        harness.completion_reason = "completed"
        harness.step = 5
        harness.action_counts = {"click": 3, "wait": 2}
        harness._git_info = {"commit": "abc123", "branch": "main", "dirty": False}

        harness._write_manifest()

        manifest_path = tmp_path / run_id / "run_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())

        assert manifest["run_id"] == run_id
        assert manifest["exit_code"] == EXIT_SUCCESS
        assert manifest["completion_reason"] == "completed"
        assert manifest["steps_completed"] == 5
        assert manifest["action_counts"] == {"click": 3, "wait": 2}
        assert manifest["git"] == {"commit": "abc123", "branch": "main", "dirty": False}
        assert manifest["game"]["type"] == sample_settings.game.type
        assert manifest["tester_version"]  # any non-empty string


# ---------------------------------------------------------------------------
# Git capture
# ---------------------------------------------------------------------------

class TestCaptureGitInfo:
    def test_non_repo_directory_returns_none_fields(self, tmp_path):
        info = capture_git_info(tmp_path)
        assert info["commit"] is None
        assert info["branch"] is None
        assert info["dirty"] is None

    def test_file_path_resolves_to_parent(self, tmp_path):
        # Should not crash when given a file path instead of a directory.
        f = tmp_path / "game.txt"
        f.write_text("hello")
        info = capture_git_info(f)
        # tmp_path is not a git repo → all None
        assert info["commit"] is None

    def test_real_repo_returns_commit(self, tmp_path):
        """Initialize a throwaway git repo and verify fields populate."""
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True
        )
        (tmp_path / "README.md").write_text("hi")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True
        )

        info = capture_git_info(tmp_path)
        assert info["commit"] is not None
        assert len(info["commit"]) == 40  # full sha
        assert info["branch"] in ("main", "master")
        assert info["dirty"] is False


# ---------------------------------------------------------------------------
# Lock behavior (module-level smoke test of the fcntl path)
# ---------------------------------------------------------------------------

class TestLockConstants:
    def test_exit_locked_constant_exists(self):
        # Just confirms we exported the constant used by cli.py
        assert EXIT_LOCKED == 4


# ---------------------------------------------------------------------------
# Integration: a full namespaced dry-run writes manifest end-to-end
# ---------------------------------------------------------------------------

class TestNamespacedDryRun:
    def test_run_writes_manifest_and_log(self, sample_settings, tmp_path):
        run_id = "integration"
        launcher = MagicMock()
        launcher.is_alive.return_value = True
        client = MagicMock(spec=LLMClient)
        # Return a 'done' action so the loop completes immediately
        client.analyze.return_value = {
            "description": "x",
            "action": "done",
            "coordinates": [],
            "reasoning": "x",
            "narrative": "x",
        }

        harness = Harness(
            sample_settings,
            launcher,
            client,
            dry_run=True,
            run_id=run_id,
            runs_dir=tmp_path,
        )
        exit_code = harness.run()

        assert exit_code == EXIT_SUCCESS
        manifest_path = tmp_path / run_id / "run_manifest.json"
        log_path = tmp_path / run_id / "playthrough_log.jsonl"
        assert manifest_path.is_file()
        assert log_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["completion_reason"] == "completed"
        assert manifest["steps_completed"] == 1
