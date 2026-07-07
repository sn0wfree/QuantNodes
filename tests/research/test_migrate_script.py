"""Tests for scripts/migrate_llmwikify_paths.py (M4.2 PR6.7).

Verifies:
  - default symlink mode (zero-copy)
  - --copy mode (physical copy)
  - --dry-run mode (no filesystem changes)
  - idempotent (existing symlinks/dirs skipped)
  - legacy ~./llmwikify doesn't exist → no-op
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.migrate_llmwikify_paths import (
    MIGRATION_PLAN,
    migrate,
)


@pytest.fixture
def fake_homes(monkeypatch, tmp_path):
    """Set HOME to tmp_path/. So ~/.llmwikify/ and ~/.quantnodes/ are isolated."""
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr("scripts.migrate_llmwikify_paths.LEGACY_HOME", fake_home / ".llmwikify")
    monkeypatch.setattr("scripts.migrate_llmwikify_paths.NEW_HOME", fake_home / ".quantnodes")
    return fake_home


class TestMigrateSymlinkMode:
    def test_symlinks_legacy_to_new(self, fake_homes) -> None:
        """symlink mode: ~/.quantnodes/akshare_cache → ~/.llmwikify/akshare_cache."""
        legacy = fake_homes / ".llmwikify"
        (legacy / "akshare_cache").mkdir(parents=True)
        (legacy / "akshare_cache" / "data.txt").write_text("hello")

        count = migrate(mode="symlink")

        assert count >= 1
        new_akshare = fake_homes / ".quantnodes" / "akshare_cache"
        assert new_akshare.is_symlink()
        assert new_akshare.resolve() == (legacy / "akshare_cache").resolve()
        assert (new_akshare / "data.txt").read_text() == "hello"

    def test_symlink_single_file(self, fake_homes) -> None:
        """symlink mode: ~/.quantnodes/llm.json → ~/.llmwikify/llmwikify.json (renamed)."""
        legacy = fake_homes / ".llmwikify"
        legacy.mkdir(parents=True)
        (legacy / "llmwikify.json").write_text('{"llm": {"model": "x"}}')

        count = migrate(mode="symlink")

        new_llm = fake_homes / ".quantnodes" / "llm.json"
        assert new_llm.is_symlink()
        assert new_llm.read_text() == '{"llm": {"model": "x"}}'

    def test_idempotent(self, fake_homes) -> None:
        """Second call returns 0 (already migrated)."""
        legacy = fake_homes / ".llmwikify"
        (legacy / "akshare_cache").mkdir(parents=True)

        migrate(mode="symlink")
        count2 = migrate(mode="symlink")
        assert count2 == 0

    def test_no_legacy_no_op(self, fake_homes) -> None:
        """When ~/.llmwikify doesn't exist, returns 0 silently."""
        count = migrate(mode="symlink")
        assert count == 0


class TestMigrateCopyMode:
    def test_copies_legacy_to_new(self, fake_homes) -> None:
        """--copy: physical copy (default: symlink)."""
        legacy = fake_homes / ".llmwikify"
        (legacy / "akshare_cache").mkdir(parents=True)
        (legacy / "akshare_cache" / "data.txt").write_text("hello")

        count = migrate(mode="copy")

        new_akshare = fake_homes / ".quantnodes" / "akshare_cache"
        assert not new_akshare.is_symlink()
        assert new_akshare.is_dir()
        assert (new_akshare / "data.txt").read_text() == "hello"


class TestMigrateDryRun:
    def test_dry_run_no_changes(self, fake_homes) -> None:
        """--dry-run: prints plan, does not modify filesystem."""
        legacy = fake_homes / ".llmwikify"
        (legacy / "akshare_cache").mkdir(parents=True)

        count = migrate(mode="symlink", dry_run=True)

        # count is still computed (planning), but no files created
        new_akshare = fake_homes / ".quantnodes" / "akshare_cache"
        assert not new_akshare.exists()
        assert count >= 1


class TestMigrateSkipsExisting:
    def test_skips_when_new_path_exists(self, fake_homes) -> None:
        """If ~/.quantnodes/X already exists (real dir), don't touch it."""
        legacy = fake_homes / ".llmwikify"
        (legacy / "akshare_cache").mkdir(parents=True)

        new = fake_homes / ".quantnodes"
        new.mkdir()
        existing = new / "akshare_cache"
        existing.mkdir()
        (existing / "user_file.txt").write_text("user data")

        count = migrate(mode="symlink")

        # Should skip because target already exists
        assert not existing.is_symlink()
        assert (existing / "user_file.txt").read_text() == "user data"


class TestMigratePlan:
    def test_plan_includes_key_paths(self) -> None:
        """Plan covers critical paths: llm.json, akshare_cache, papers, etc."""
        new_subs = {entry[0] for entry in MIGRATION_PLAN}
        assert "llm.json" in new_subs
        assert "akshare_cache" in new_subs
        assert "reproduction.db" in new_subs
        assert "papers" in new_subs
        assert "strategies" in new_subs


class TestMigrateCLI:
    """Integration: subprocess invocation."""

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.migrate_llmwikify_paths", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/ll/Public/QuantNodes",
        )
        assert result.returncode == 0
        assert "--copy" in result.stdout
        assert "--dry-run" in result.stdout

    def test_cli_dry_run(self, tmp_path) -> None:
        env = os.environ.copy()
        env["HOME"] = str(tmp_path)
        (tmp_path / ".llmwikify" / "akshare_cache").mkdir(parents=True)
        result = subprocess.run(
            [sys.executable, "-m", "scripts.migrate_llmwikify_paths", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            cwd="/home/ll/Public/QuantNodes",
        )
        assert result.returncode == 0
        assert "[DRY]" in result.stdout or "[done]" in result.stdout