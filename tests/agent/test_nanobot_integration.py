# coding=utf-8
"""End-to-end integration tests for QuantNodes Agent (nanobot bridge).

Tests:
- Agent(workspace) constructor builds Nanobot + 14 quant tools
- Agent.run_stream emits v2.x token/tool_call/tool_result/done events
- nanobot_config.json is written correctly by config_mapper
- API services wire to Agent via nanobot_bridge (settings reload)
"""

import json
import os
from pathlib import Path

import pytest

from QuantNodes.agent import Agent
from QuantNodes.agent.config_mapper import build_nanobot_config, write_nanobot_config


@pytest.fixture
def fake_api_key(monkeypatch):
    """Set a dummy API key for tests that exercise the provider factory."""
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "sk-test-dummy-key-12345")
    monkeypatch.setenv("QUANTNODES__LLM__BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("QUANTNODES__LLM__MODEL", "gpt-4o")


@pytest.fixture
def fresh_workspace(tmp_path: Path) -> Path:
    """Empty workspace under tmp_path/.agent/."""
    ws = tmp_path / ".agent"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_agent_constructs_with_nanobot(fake_api_key, fresh_workspace):
    """Agent(workspace) should succeed and load Nanobot + quant tools."""
    agent = Agent(workspace=str(fresh_workspace))
    assert agent.workspace == fresh_workspace.resolve()
    assert (fresh_workspace / "nanobot_config.json").exists()
    assert agent.loop is not None


def test_agent_default_workspace_is_dot_agent(fake_api_key, monkeypatch, tmp_path):
    """If no workspace is given, default to ``.agent/`` (HKUDS nanobot convention)."""
    monkeypatch.chdir(tmp_path)
    agent = Agent()
    assert str(agent.workspace).endswith(".agent")


def test_agent_registers_14_quant_tools(fake_api_key, fresh_workspace):
    """14 quant tools must appear in nanobot's ToolRegistry."""
    agent = Agent(workspace=str(fresh_workspace))
    names = set(agent.loop.tools._tools.keys())
    expected = {
        "echo", "sandbox", "pipeline", "strategy", "backtest",
        "factor", "config_backtest", "wiki", "file_ops",
        "code_search", "git_ops", "web_fetch", "web_search", "task",
    }
    missing = expected - names
    assert not missing, f"missing quant tools: {missing}"


def test_config_mapper_uses_custom_slot_for_openai_compat(fake_api_key, fresh_workspace):
    """Generic OpenAI-compatible URL → ``providers.custom`` slot."""
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "custom"
    assert "custom" in cfg["providers"]
    assert cfg["providers"]["custom"]["api_key"] == "sk-test-dummy-key-12345"


def test_config_mapper_anthropic_key_routes_to_anthropic(monkeypatch, fresh_workspace):
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "sk-ant-test-12345")
    monkeypatch.delenv("QUANTNODES__LLM__BASE_URL", raising=False)
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "anthropic"


def test_config_mapper_ollama_url_routes_to_ollama(monkeypatch, fresh_workspace):
    monkeypatch.setenv("QUANTNODES__LLM__API_KEY", "ollama")
    monkeypatch.setenv("QUANTNODES__LLM__BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("QUANTNODES__LLM__MODEL", raising=False)
    cfg = build_nanobot_config(fresh_workspace)
    assert cfg["agents"]["defaults"]["provider"] == "ollama"


def test_write_nanobot_config_creates_file(tmp_path):
    ws = tmp_path / ".agent"
    cfg = {"agents": {"defaults": {"workspace": str(ws)}}, "providers": {}}
    target = write_nanobot_config(ws, cfg)
    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["agents"]["defaults"]["workspace"] == str(ws)


def test_agent_config_json_contains_expected_keys(fake_api_key, fresh_workspace):
    """config.json written by Agent must satisfy upstream schema (no validation errors)."""
    agent = Agent(workspace=str(fresh_workspace))
    cfg = json.loads((fresh_workspace / "nanobot_config.json").read_text())
    assert "agents" in cfg
    assert "providers" in cfg
    # upstream requires these top-level keys under agents.defaults
    assert "model" in cfg["agents"]["defaults"]
    assert "provider" in cfg["agents"]["defaults"]


@pytest.mark.asyncio
async def test_agent_loop_attribute_exists(fake_api_key, fresh_workspace):
    """Agent.loop should expose the upstream AgentLoop with tools attribute."""
    agent = Agent(workspace=str(fresh_workspace))
    assert hasattr(agent.loop, "tools")
    assert hasattr(agent.loop, "provider")


def test_agent_settings_reload_no_leak(fake_api_key, fresh_workspace):
    """Constructing Agent twice should not leak config files into workspace."""
    Agent(workspace=str(fresh_workspace))
    before = set(p.name for p in fresh_workspace.iterdir())
    Agent(workspace=str(fresh_workspace))
    after = set(p.name for p in fresh_workspace.iterdir())
    assert before == after, f"workspace files changed: {before ^ after}"
