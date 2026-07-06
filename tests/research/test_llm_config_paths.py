"""Tests for LLM config path resolution (M3.2).

Covers:
  - CONFIG_PATHS priority: ~/.quantnodes/llm.json > ~/.llmwikify/llmwikify.json
  - QUANTNODES__LLM__* env var overrides
  - Empty / missing config graceful fallback
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from QuantNodes.research.common.llm.client import (
    CONFIG_PATHS,
    _apply_env_overrides,
    _load_single_path,
    build_llm_client,
    load_llm_config,
)


class TestLoadSinglePath:
    """Unit tests for _load_single_path helper."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = _load_single_path(tmp_path / "nonexistent.json")
        assert result is None

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("{}", encoding="utf-8")
        result = _load_single_path(f)
        assert result == {}

    def test_no_llm_section_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "no_llm.json"
        f.write_text(json.dumps({"other": {"key": "val"}}), encoding="utf-8")
        result = _load_single_path(f)
        assert result == {}

    def test_valid_llm_section(self, tmp_path: Path) -> None:
        f = tmp_path / "good.json"
        f.write_text(json.dumps({
            "llm": {"enabled": True, "model": "test"}
        }), encoding="utf-8")
        result = _load_single_path(f)
        assert result == {"enabled": True, "model": "test"}

    def test_invalid_json_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json!", encoding="utf-8")
        result = _load_single_path(f)
        assert result == {}

    def test_non_dict_top_level_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "list.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        result = _load_single_path(f)
        assert result == {}


class TestLoadLlmConfigPriority:
    """Priority: explicit config_path > CONFIG_PATHS[0] > CONFIG_PATHS[1] > {}."""

    def test_explicit_config_path_used(self, tmp_path: Path, monkeypatch) -> None:
        """Explicit config_path wins over everything."""
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({
            "llm": {"model": "explicit", "api_key": "k1"}
        }), encoding="utf-8")
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (tmp_path / "never.json", tmp_path / "never2.json"),
        )
        assert load_llm_config(config_path=explicit)["model"] == "explicit"

    def test_first_path_priority(self, tmp_path: Path, monkeypatch) -> None:
        """CONFIG_PATHS[0] (new) wins when both exist."""
        new = tmp_path / "new.json"
        legacy = tmp_path / "legacy.json"
        new.write_text(json.dumps({
            "llm": {"model": "new-model", "api_key": "k1"}
        }), encoding="utf-8")
        legacy.write_text(json.dumps({
            "llm": {"model": "legacy-model", "api_key": "k2"}
        }), encoding="utf-8")
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (new, legacy),
        )
        result = load_llm_config()
        assert result["model"] == "new-model"

    def test_fallback_to_second_path(self, tmp_path: Path, monkeypatch) -> None:
        """When first path missing, falls back to second."""
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({
            "llm": {"model": "legacy-model", "api_key": "k2"}
        }), encoding="utf-8")
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (tmp_path / "nonexistent.json", legacy),
        )
        result = load_llm_config()
        assert result["model"] == "legacy-model"

    def test_empty_config_returns_empty_dict(self, tmp_path: Path, monkeypatch) -> None:
        """Empty llm section returns {}."""
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"llm": {}}), encoding="utf-8")
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (f, tmp_path / "also_empty.json"),
        )
        # {llm: {}} is found but empty — still treated as "found"
        result = load_llm_config()
        assert result == {}

    def test_no_paths_found_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        """All paths missing → {}."""
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (tmp_path / "a.json", tmp_path / "b.json"),
        )
        assert load_llm_config() == {}


class TestApplyEnvOverrides:
    """QUANTNODES__LLM__* env vars override config values."""

    def test_env_sets_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__PROVIDER", "openai")
        config = {"provider": "minimax"}
        result = _apply_env_overrides(config)
        assert result["provider"] == "openai"

    def test_env_sets_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "env-key-123")
        config = {"api_key": "file-key"}
        result = _apply_env_overrides(config)
        assert result["api_key"] == "env-key-123"

    def test_env_sets_model(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__MODEL", "env-model")
        config = {"model": "file-model"}
        result = _apply_env_overrides(config)
        assert result["model"] == "env-model"

    def test_env_sets_base_url(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__BASE_URL", "https://env.api.com/v1")
        config = {"base_url": "https://file.api.com/v1"}
        result = _apply_env_overrides(config)
        assert result["base_url"] == "https://env.api.com/v1"

    def test_env_enabled_coerces_bool(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__ENABLED", "true")
        result = _apply_env_overrides({})
        assert result["enabled"] is True

    def test_env_enabled_false(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__ENABLED", "false")
        result = _apply_env_overrides({})
        assert result["enabled"] is False

    def test_empty_env_ignored(self, monkeypatch) -> None:
        """Empty env values are ignored (config preserved)."""
        monkeypatch.delenv("QUANTNODES__LLM__PROVIDER", raising=False)
        config = {"provider": "minimax"}
        result = _apply_env_overrides(config)
        assert result["provider"] == "minimax"

    def test_does_not_mutate_input(self, monkeypatch) -> None:
        monkeypatch.setenv("QUANTNODES__LLM__MODEL", "new")
        original = {"model": "old"}
        _apply_env_overrides(original)
        assert original["model"] == "old"


class TestBuildLlmClientWithEnv:
    """build_llm_client integrates env overrides."""

    def test_env_overrides_file_config(self, tmp_path: Path, monkeypatch) -> None:
        """QUANTNODES__LLM__MODEL overrides file value in build_llm_client."""
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({
            "llm": {
                "enabled": True,
                "provider": "minimax",
                "model": "file-model",
                "api_key": "test-key",
                "base_url": "https://api.minimaxi.com/v1",
            }
        }), encoding="utf-8")
        monkeypatch.setenv("QUANTNODES__LLM__MODEL", "env-model")
        monkeypatch.setattr(
            "QuantNodes.research.common.llm.client.CONFIG_PATHS",
            (config_file,),
        )
        client = build_llm_client(config_path=config_file)
        assert client.model == "env-model"
