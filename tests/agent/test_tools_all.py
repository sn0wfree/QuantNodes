# coding=utf-8
"""Tests for v3.0.0 ``nanobot.agent.runner.AgentRunner`` + ``AgentLoop`` (mocked).

v3.0.0 refactor: Stage 1 replaced the local ``QuantNodes.agent.core.loop.AgentLoop``
and ``QuantNodes.agent.core.runner.AgentRunner`` (v2.x) with the upstream
HKUDS nanobot versions:

- ``nanobot.agent.runner.AgentRunner`` — runs an agent turn (LLM call +
  tool dispatch + result integration), driven by ``AgentRunSpec``.
- ``nanobot.agent.loop.AgentLoop`` — the long-lived runtime that
  consumes from a ``MessageBus`` and dispatches turns.

Both are exercised here via a ``MockProvider`` (a stand-in
``LLMProvider``) and ``MockAgentLoop`` (a stand-in for the runtime).
This is the v3.0.0 strategy: don't depend on a real LLM; mock at the
``LLMProvider`` / ``AgentLoop`` boundary.

All tests in this file are **skipped when ``nanobot-ai`` is not
installed** because the upstream modules aren't available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from QuantNodes.agent import NANOBOT_AVAILABLE


# Skip the entire module when nanobot-ai is not installed — the
# upstream modules we exercise here don't exist in the no-extras case.
pytestmark = pytest.mark.skipif(
    not NANOBOT_AVAILABLE,
    reason="AgentRunner/AgentLoop tests require nanobot-ai (Stage 5.3 graceful degradation)",
)


# ----------------------------------------------------------------------------
# Mock LLMProvider
# ----------------------------------------------------------------------------

class MockProvider:
    """Stand-in for upstream ``LLMProvider`` — returns scripted responses.

    Implements the v3.0.0 contract: a ``chat()`` method that returns
    an object with ``content``, ``tool_calls``, and ``finish_reason``
    attributes. Mirrors the upstream ``LLMResponse`` shape (we use
    ``MagicMock`` to avoid a direct import of nanobot here).
    """

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.call_count = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        # default: stop with empty content
        return _make_response(content="Done", finish_reason="stop")


def _make_response(content=None, tool_calls=None, finish_reason="stop"):
    """Build a response object mimicking upstream ``LLMResponse``."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    resp.finish_reason = finish_reason
    return resp


def _make_tool_call(id: str, name: str, arguments: Dict[str, Any]):
    """Build a ``ToolCallRequest``-shaped object."""
    tc = MagicMock()
    tc.id = id
    tc.name = name
    tc.arguments = arguments
    return tc


# ----------------------------------------------------------------------------
# AgentRunner tests
# ----------------------------------------------------------------------------

class TestAgentRunner:
    """``nanobot.agent.runner.AgentRunner`` — turn-level execution."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.agent.tools.registry import ToolRegistry

        provider = MockProvider([
            _make_response(content="Hello!", finish_reason="stop"),
        ])
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=ToolRegistry(),
        )
        result = await runner.run(spec)
        assert result.final_content == "Hello!"
        assert result.stop_reason in ("completed", "stop", "max_iterations")
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_execution(self):
        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.agent.tools.registry import ToolRegistry

        # Register EchoTool upstream-side (we register against the
        # *upstream* ToolRegistry, not our local one).
        from QuantNodes.agent.tools import EchoTool
        tool_reg = ToolRegistry()
        tool_reg.register(EchoTool())

        provider = MockProvider([
            # First call: ask for tool
            _make_response(
                content=None,
                tool_calls=[_make_tool_call("tc1", "echo", {"message": "hello"})],
                finish_reason="tool_calls",
            ),
            # Second call: final
            _make_response(content="Echo returned: hello", finish_reason="stop"),
        ])
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "echo hello"}],
            tools=tool_reg,
            max_iterations=3,
        )
        result = await runner.run(spec)
        assert result.final_content == "Echo returned: hello"
        # Provider was called twice (tool_call → final)
        assert provider.call_count == 2
        # Tools used includes echo
        used = " ".join(getattr(result, "tools_used", []) or [])
        assert "echo" in used or "Echo" in used

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.agent.tools.registry import ToolRegistry

        # Provider always returns a tool_call → infinite loop unless
        # max_iterations caps it.
        provider = MockProvider([
            _make_response(
                tool_calls=[_make_tool_call("tc", "echo", {"message": "loop"})],
                finish_reason="tool_calls",
            )
        ] * 10)
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "loop"}],
            tools=ToolRegistry(),
            max_iterations=2,
        )
        result = await runner.run(spec)
        # Runner must stop at max_iterations, not call LLM forever
        assert provider.call_count <= 3  # initial + a couple iterations
        assert result.stop_reason in ("max_iterations", "max_tool_iterations")


# ----------------------------------------------------------------------------
# AgentLoop tests
# ----------------------------------------------------------------------------

class TestAgentLoop:
    """``nanobot.agent.loop.AgentLoop`` — long-lived runtime with bus."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory(prefix="test_loop_") as tmp:
            yield Path(tmp)

    @pytest.mark.asyncio
    async def test_chat_returns_text_response(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        provider = MockProvider([
            _make_response(content="I'm your quant assistant.", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace=workspace_path(workspace))
        result = await loop.chat("你好", session_id="test")
        assert result == "I'm your quant assistant."

    @pytest.mark.asyncio
    async def test_chat_with_tool_dispatch(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus
        from QuantNodes.agent.tools import EchoTool

        provider = MockProvider([
            _make_response(
                tool_calls=[_make_tool_call("tc1", "echo", {"message": "test"})],
                finish_reason="tool_calls",
            ),
            _make_response(content="Echo returned: test", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace=workspace_path(workspace))
        loop.register_tool(EchoTool())
        result = await loop.chat("echo test", session_id="test_tool")
        assert "test" in result

    @pytest.mark.asyncio
    async def test_session_persistence_across_chats(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        provider = MockProvider([
            _make_response(content="Reply 1", finish_reason="stop"),
            _make_response(content="Reply 2", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace=workspace_path(workspace))
        await loop.chat("Q1", session_id="persist")
        await loop.chat("Q2", session_id="persist")
        session = loop.session_manager.get_session("persist")
        # 2 user + 2 assistant = 4 messages
        assert len(session.messages) == 4

    @pytest.mark.asyncio
    async def test_different_sessions_isolated(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        provider = MockProvider([
            _make_response(content="A reply", finish_reason="stop"),
            _make_response(content="B reply", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace=workspace_path(workspace))
        await loop.chat("A question", session_id="session_a")
        await loop.chat("B question", session_id="session_b")
        sa = loop.session_manager.get_session("session_a")
        sb = loop.session_manager.get_session("session_b")
        assert sa.messages[0]["content"] == "A question"
        assert sb.messages[0]["content"] == "B question"


def workspace_path(tmp_path: Path) -> str:
    """Convert tmp_path to a string path (upstream AgentLoop expects str)."""
    return str(tmp_path)
