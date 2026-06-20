"""Tests for core/path_utils.py — Phase G1."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from QuantNodes.core.path_utils import ensure_dir, ensure_parent, resolve_path


class TestResolvePath:
    def test_default_only(self) -> None:
        result = resolve_path("~/foo/bar")
        assert result == Path("~/foo/bar").expanduser()

    def test_env_wins_over_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("QN_TEST_OUT", str(tmp_path / "from_env"))
        result = resolve_path("~/default", env_var="QN_TEST_OUT")
        assert result == tmp_path / "from_env"

    def test_empty_env_falls_through_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QN_TEST_OUT", "")
        result = resolve_path("~/default", env_var="QN_TEST_OUT")
        assert result == Path("~/default").expanduser()

    def test_no_env_var(self) -> None:
        result = resolve_path("~/abc")
        assert isinstance(result, Path)
        assert "~" not in str(result)


class TestEnsureDir:
    def test_creates_new(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_idempotent(self, tmp_path: Path) -> None:
        ensure_dir(tmp_path / "x")
        ensure_dir(tmp_path / "x")
        assert (tmp_path / "x").is_dir()

    def test_expanduser(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            os.environ["HOME"] = str(home)
            target = ensure_dir("~/qd")
            assert target.is_dir()
            assert target == home / "qd"


class TestEnsureParent:
    def test_creates_parent(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "file.txt"
        result = ensure_parent(target)
        assert target.parent.is_dir()
        assert result == target

    def test_idempotent(self, tmp_path: Path) -> None:
        ensure_parent(tmp_path / "y" / "file.txt")
        ensure_parent(tmp_path / "y" / "file.txt")
        assert (tmp_path / "y").is_dir()
