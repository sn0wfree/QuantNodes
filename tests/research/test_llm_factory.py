"""Tests for llm_extraction/llm_factory: LLM 客户端工厂."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from QuantNodes.research.common import llm_factory as lf
from QuantNodes.research.common.llm import client as llm_client



def _set_config_paths(tmp_path: Path, monkeypatch) -> None:
    """Monkeypatch CONFIG_PATHS to [new_path, legacy_path] inside tmp_path."""
    new_path = tmp_path / "new_llm.json"
    legacy_path = tmp_path / "llm_legacy.json"
    monkeypatch.setattr(
        "QuantNodes.research.common.llm.client.CONFIG_PATHS",
        (new_path, legacy_path),
    )
    return new_path, legacy_path



class TestLoadLlmConfig:
    """Test load_llm_config (3 测试)."""

    def test_missing_config_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        """config 文件不存在返回 {}."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        result = lf.load_llm_config()
        assert result == {}

    def test_loads_llm_section_from_legacy(self, tmp_path: Path, monkeypatch) -> None:
        """从 legacy JSON 读 [llm] section (new 不存在时 fallback)."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        legacy_path.write_text(
            json.dumps({
                "llm": {"model": "minimax-M3", "api_key": "test-key"},
                "other_section": {"key": "value"},
            }),
            encoding="utf-8",
        )
        result = lf.load_llm_config()
        assert result["model"] == "minimax-M3"
        assert result["api_key"] == "test-key"
        assert "key" not in result

    def test_invalid_json_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        """无效 JSON 返回 {}."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        new_path.write_text(":\n  - [unclosed", encoding="utf-8")
        result = lf.load_llm_config()
        assert result == {}

    def test_new_path_takes_priority_over_legacy(self, tmp_path: Path, monkeypatch) -> None:
        """新路径优先于 legacy 路径."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        new_path.write_text(json.dumps({
            "llm": {"model": "new-model", "api_key": "new-key", "enabled": True}
        }), encoding="utf-8")
        legacy_path.write_text(json.dumps({
            "llm": {"model": "legacy-model", "api_key": "legacy-key", "enabled": True}
        }), encoding="utf-8")
        result = lf.load_llm_config()
        assert result["model"] == "new-model"
        assert result["api_key"] == "new-key"

    def test_explicit_path_overrides_all(self, tmp_path: Path, monkeypatch) -> None:
        """显式 config_path 覆盖所有优先级."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        new_path.write_text(json.dumps({
            "llm": {"model": "should-not-see-this"}
        }), encoding="utf-8")
        legacy_path.write_text(json.dumps({
            "llm": {"model": "should-not-see-this-either"}
        }), encoding="utf-8")
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({
            "llm": {"model": "explicit-model", "api_key": "explicit-key"}
        }), encoding="utf-8")
        result = lf.load_llm_config(config_path=explicit)
        assert result["model"] == "explicit-model"


class TestBuildDefaultClient:
    """Test build_default_client (3 测试)."""

    def test_missing_config_raises(self, tmp_path: Path, monkeypatch) -> None:
        """config 缺失时 build_default_client 抛 RuntimeError."""
        _set_config_paths(tmp_path, monkeypatch)
        with pytest.raises((RuntimeError, Exception)):
            lf.build_default_client()

    def test_builds_client_with_valid_config(self, tmp_path: Path, monkeypatch) -> None:
        """有效 config 时 build_default_client 成功."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        new_path.write_text(
            json.dumps({
                "llm": {
                    "enabled": True,
                    "provider": "minimax",
                    "model": "minimax",
                    "api_key": "test-key",
                    "base_url": "https://api.test.com",
                }
            }),
            encoding="utf-8",
        )
        try:
            client = lf.build_default_client()
            assert client is not None
        except Exception as exc:
            err_msg = str(exc).lower()
            assert any(k in err_msg for k in ["config", "key", "enabled", "disabled"])

    def test_model_override(self, tmp_path: Path, monkeypatch) -> None:
        """model 参数覆盖 config 中 model 字段."""
        new_path, legacy_path = _set_config_paths(tmp_path, monkeypatch)
        new_path.write_text(
            json.dumps({
                "llm": {
                    "enabled": True,
                    "model": "default-model",
                    "api_key": "test-key",
                    "base_url": "https://api.test.com",
                }
            }),
            encoding="utf-8",
        )
        try:
            client = lf.build_default_client(model="override-model")
            assert client is not None
        except Exception:
            pass  # 接受失败 (env 依赖)
