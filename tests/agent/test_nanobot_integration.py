# coding=utf-8
"""End-to-end integration tests for QuantNodes Agent (nanobot bridge).

Stage 1-5: This file was refactored in v3.0.0 to remove tests that
target the v2.x ``QuantNodes.agent.core.loop`` / ``runner`` / ``memory``
modules — all of which were deleted in Stage 1 when we migrated to
upstream ``HKUDS/nanobot`` (see ``docs/14-上游nanobot升级指南.md``).

What's left in this file:

- 3 ``config_mapper`` tests that exercise the JSON dict produced by
  ``build_nanobot_config`` — these run **without** ``nanobot-ai`` and
  validate pure-Python dict structure.
- 1 ``write_nanobot_config`` test for the file writer.
- A ``fake_api_key`` fixture (sets ``QUANTNODES__LLM__*`` env vars) for
  future tests that need them.

Tests that exercise ``Agent(workspace)`` (i.e. instantiate the
nanobot-backed runtime) live in ``tests/agent/test_nanobot_runtime.py``
where they can use ``unittest.mock`` to bypass the nanobot-ai
installation requirement and exercise the build/schedule logic
deterministically.
"""

import json

import pytest

from QuantNodes.agent.config_mapper import build_nanobot_config, write_nanobot_config


@pytest.fixture
def fake_api_key(monkeypatch):
    """Set a dummy API key for tests that exercise the provider factory."""
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "sk-test-dummy-key-12345")
    monkeypatch.setenv("QUANTNODES__LLM__BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("QUANTNODES__LLM__MODEL", "gpt-4o")


@pytest.fixture
def fresh_workspace(tmp_path):
    """Empty workspace under ``tmp_path/.agent/``."""
    from pathlib import Path

    ws = tmp_path / ".agent"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# ----------------------------------------------------------------------------
# config_mapper — pure-dict tests (no nanobot-ai required)
# ----------------------------------------------------------------------------

def test_config_mapper_uses_custom_slot_for_openai_compat(fake_api_key, fresh_workspace):
    """Generic OpenAI-compatible URL → ``providers.custom`` slot.

    Stage 5.4: config_mapper writes fields in camelCase (matching
    upstream nanobot's ``Config.model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)``).
    The dict is loaded via Pydantic which accepts both ``api_key`` and
    ``apiKey`` as aliases; tests that read the raw JSON must use camelCase.
    """
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "custom"
    assert "custom" in cfg["providers"]
    # Field name is camelCase in the raw JSON (Stage 5.4 change)
    assert cfg["providers"]["custom"]["apiKey"] == "sk-test-dummy-key-12345"


def test_config_mapper_anthropic_key_routes_to_anthropic(monkeypatch, fresh_workspace):
    """``sk-ant-*`` API key → ``providers.anthropic`` slot, no base URL needed."""
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "sk-ant-test-12345")
    monkeypatch.delenv("QUANTNODES__LLM__BASE_URL", raising=False)
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "anthropic"


def test_config_mapper_ollama_url_routes_to_ollama(monkeypatch, fresh_workspace):
    """Ollama-style URL (``localhost:11434``) → ``providers.ollama`` slot."""
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "ollama")
    monkeypatch.setenv("QUANTNODES__LLM__BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("QUANTNODES__LLM__MODEL", raising=False)
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "ollama"


def test_write_nanobot_config_creates_file(tmp_path):
    """``write_nanobot_config`` persists the dict to ``<ws>/nanobot_config.json``."""
    ws = tmp_path / ".agent"
    cfg = {"agents": {"defaults": {"workspace": str(ws)}}, "providers": {}}
    target = write_nanobot_config(ws, cfg)
    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["agents"]["defaults"]["workspace"] == str(ws)
