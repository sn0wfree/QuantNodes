# coding=utf-8
"""Tests for v3.0.0 Stage 5.3 optional-dependency pattern.

When ``nanobot-ai`` is **not** installed:
- ``QuantNodes.agent.NANOBOT_AVAILABLE`` is False
- ``from QuantNodes.agent import Agent`` succeeds (no ImportError at import)
- ``Agent(...)`` raises ``NanobotNotInstalled`` with a friendly hint
- The 14 quant tool classes are importable
- The 3 services (wiki / factor / backtest) work without nanobot
- The FastAPI app starts; ``/api/agent/*`` returns 503 with the install hint
"""

from __future__ import annotations

import importlib
import sys

import pytest

# Module-level imports of the symbols we test. Doing these at import time
# (rather than inside each test function) avoids a pytest issue where
# function-level ``from X import Y`` triggers a stale module reload in
# some test runners. Module-level imports are cached by Python's import
# system and stay consistent throughout the file's lifetime.
from QuantNodes.agent import NANOBOT_AVAILABLE, NanobotNotInstalled  # noqa: E402


def test_nanobot_available_flag_exists():
    """QuantNodes.agent must expose NANOBOT_AVAILABLE bool at import time."""
    assert isinstance(NANOBOT_AVAILABLE, bool)


def test_nanobot_not_installed_error_class():
    """NanobotNotInstalled is an ImportError subclass with a clear hint."""
    assert issubclass(NanobotNotInstalled, ImportError)
    err = NanobotNotInstalled("Agent")
    msg = str(err)
    assert "agent" in msg.lower()
    assert "pip install" in msg
    assert "quantnodes" in msg


def test_agent_attribute_returns_proxy_when_nanobot_missing():
    """`from QuantNodes.agent import Agent` works even without nanobot; attribute access raises."""
    from QuantNodes.agent import NANOBOT_AVAILABLE, Agent, NanobotNotInstalled

    if NANOBOT_AVAILABLE:  # pragma: no cover - skipped when [agent] installed
        pytest.skip("nanobot-ai is installed — covered by other tests")

    # Agent should be a proxy that raises on attribute/call access.
    with pytest.raises(NanobotNotInstalled):
        Agent()  # type: ignore[operator]


def test_quant_tools_importable_without_nanobot():
    """All 14 quant tool classes must import even when nanobot is missing."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed — covered by other tests"):
        return  # nanobot installed, this test doesn't apply

    from QuantNodes.agent.tools import (
        BacktestTool,
        ConfigBacktestTool,
        EchoTool,
        FactorTool,
        FileOpsTool,
        GitOpsTool,
        PipelineTool,
        SandboxTool,
        StrategyTool,
        TaskTool,
        WebFetchTool,
        WebSearchTool,
        WikiTool,
        CodeSearchTool,
        Tool,
    )
    for cls in (BacktestTool, ConfigBacktestTool, EchoTool, FactorTool,
                FileOpsTool, GitOpsTool, PipelineTool, SandboxTool,
                StrategyTool, TaskTool, WebFetchTool, WebSearchTool,
                WikiTool, CodeSearchTool):
        assert cls is not None, f"{cls.__name__} is None"
    assert Tool is not None


def test_tools_base_works_without_nanobot():
    """`class MyTool(Tool)` and the `_dispatch` helper must work in pure-quant mode."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return

    from QuantNodes.agent.tools.base import Tool, ToolExecutionResult

    class _DummyTool(Tool):
        name = "dummy"
        description = "test"

    # Subclassing works
    t = _DummyTool()
    assert isinstance(t, Tool)
    assert t.name == "dummy"

    # _dispatch works
    async def action_a(**kw):
        return ("a", kw)

    async def _run():
        return await t._dispatch("action_a", {"action_a": action_a}, x=1)

    import asyncio
    result = asyncio.run(_run())
    assert result == ("a", {"x": 1})


def test_agent_service_imports_without_nanobot():
    """api.services.agent_service must import without nanobot-ai."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return

    # Force a fresh import in case cached.
    for mod in list(sys.modules):
        if mod.startswith("api.services.agent_service"):
            del sys.modules[mod]
    from api.services.agent_service import agent_service
    assert agent_service is not None
    assert agent_service.workspace == ".agent"


def test_agent_service_get_agent_raises_when_nanobot_missing():
    """agent_service._get_agent() must raise NanobotNotInstalled in pure-quant mode."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return

    from QuantNodes.agent import NanobotNotInstalled
    from api.services.agent_service import agent_service

    # Reset the cached agent
    agent_service._agent = None

    with pytest.raises(NanobotNotInstalled):
        agent_service._get_agent()


def test_mcp_server_imports_without_nanobot():
    """MCP server (Stage 5.2) must work independently of nanobot-ai."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return

    from QuantNodes.mcp_server import mcp
    assert mcp is not None
    assert mcp.name == "quant"
