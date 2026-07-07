"""Tests for QuantNodes.research.common.paths (M4.2 PR6.7).

Verifies:
  - QUANTNODES_HOME = ~/.quantnodes (single source of truth)
  - quantnodes_path("sub") returns ~/.quantnodes/<sub>
  - ensure_migrated creates parent dirs (idempotent)
  - No legacy ~/.llmwikify fallback (M4.2 hardcoded)
"""
from __future__ import annotations

import pytest

from QuantNodes.research.common.paths import (
    QUANTNODES_HOME,
    ensure_migrated,
    quantnodes_path,
)


class TestQuantNodesHome:
    def test_home_under_user_home(self) -> None:
        assert QUANTNODES_HOME.name == ".quantnodes"
        assert QUANTNODES_HOME.parent == pytest.importorskip("pathlib").Path.home()


class TestPath:
    def test_simple_subpath(self) -> None:
        result = quantnodes_path("llm.json")
        assert result == QUANTNODES_HOME / "llm.json"

    def test_nested_subpath(self) -> None:
        result = quantnodes_path("akshare_cache/quantnodes_h5")
        assert result == QUANTNODES_HOME / "akshare_cache" / "quantnodes_h5"

    def test_path_returns_pathlib(self) -> None:
        from pathlib import Path
        assert isinstance(quantnodes_path("llm.json"), Path)


class TestEnsureMigrated:
    def test_creates_parent_dir(self, tmp_path, monkeypatch) -> None:
        """ensure_migrated creates missing parent dirs."""
        monkeypatch.setattr(
            "QuantNodes.research.common.paths.QUANTNODES_HOME",
            tmp_path,
        )
        target = ensure_migrated("akshare_cache")
        assert target == tmp_path / "akshare_cache"
        assert target.exists()

    def test_idempotent_on_existing_dir(self, tmp_path, monkeypatch) -> None:
        """Calling twice does not raise."""
        monkeypatch.setattr(
            "QuantNodes.research.common.paths.QUANTNODES_HOME",
            tmp_path,
        )
        ensure_migrated("cache")
        ensure_migrated("cache")
        assert (tmp_path / "cache").exists()

    def test_file_subpath_creates_parent(self, tmp_path, monkeypatch) -> None:
        """File subpath: parent dir created, file not created."""
        monkeypatch.setattr(
            "QuantNodes.research.common.paths.QUANTNODES_HOME",
            tmp_path,
        )
        target = ensure_migrated("akshare_cache/sub/file.json")
        parent = tmp_path / "akshare_cache" / "sub"
        assert parent.exists()
        assert not target.exists()  # caller creates file

    def test_no_failure_on_permission_error(self, tmp_path, monkeypatch) -> None:
        """Logs warning, does not raise on OSError."""
        monkeypatch.setattr(
            "QuantNodes.research.common.paths.QUANTNODES_HOME",
            tmp_path / "readonly",
        )
        # Make parent un-creatable
        (tmp_path / "readonly").mkdir()
        (tmp_path / "readonly" / "akshare_cache").write_text("file")
        # ensure_migrated should not raise even if mkdir fails
        # (it logs warning, returns target path)
        result = ensure_migrated("akshare_cache/x.json")
        assert result == tmp_path / "readonly" / "akshare_cache" / "x.json"