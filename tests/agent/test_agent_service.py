# coding=utf-8
"""Tests for ``api.services.agent_service.AgentService`` with a mocked agent loop.

v3.0.0 refactor: Stage 1 replaced the v2.x local ``AgentLoop``
(``QuantNodes.agent.core.loop.AgentLoop``) with the upstream
``nanobot.agent.loop.AgentLoop``. ``api.services.agent_service``
wraps the v3.0.0 ``Agent`` facade (which exposes the upstream
``loop`` attribute).

These tests inject a **minimal ``MockAgentLoop``** that:

- Exposes the same ``session_manager.get_session(...)`` surface as
  the upstream ``AgentLoop``
- Returns scripted content from ``chat()`` (the async coroutine that
  ``AgentService.send_message`` calls)
- Records the ``last_messages`` it received so tests can verify
  message-count and de-duplication

All AgentService behavior is tested **without** the nanobot runtime —
this is the v3.0.0 strategy: mock at the ``Agent`` facade boundary.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from api.services.agent_service import AgentService


# ----------------------------------------------------------------------------
# Mock agent loop
# ----------------------------------------------------------------------------

class MockSessionManager:
    def __init__(self, sessions: Dict[str, Dict[str, Any]]) -> None:
        self._sessions = sessions

    def get_session(self, session_id: str) -> Any:
        sess = self._sessions.setdefault(session_id, {"messages": []})
        return _SessionView(sess)

    def list_sessions_with_info(self) -> List[Dict[str, Any]]:
        """Stand-in for the v2.x upstream session listing.

        Returns one entry per session with the fields the v2.x
        ``AgentService.list_sessions`` expected.
        """
        out: List[Dict[str, Any]] = []
        for sid, sess in self._sessions.items():
            out.append({
                "session_id": sid,
                "message_count": len(sess.get("messages", [])),
                "created_at": "",
                "updated_at": "",
            })
        return out


class _SessionView:
    """Wraps a session dict to expose ``.messages`` like upstream Session."""

    def __init__(self, sess: Dict[str, Any]) -> None:
        self._sess = sess

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._sess.setdefault("messages", [])


class MockAgent:
    """Stand-in for the v3.0.0 ``Agent`` facade used by ``AgentService``.

    The service calls ``self._agent.run(content, session_id)``, NOT
    ``self._agent.loop.chat(...)`` directly. So the mock exposes a
    ``run()`` coroutine that records each call into the session and
    returns the scripted response.

    Also exposes a ``loop`` attribute (pointing at this object) for the
    ``session_manager`` / history APIs.
    """

    def __init__(self, response: str = "Mock response") -> None:
        self._response = response
        self.call_count = 0
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.last_messages: List[Dict[str, Any]] | None = None

    @property
    def loop(self) -> Any:
        return self

    @property
    def session_manager(self) -> MockSessionManager:
        return MockSessionManager(self.sessions)

    async def run(self, content: str, session_id: str = "default", **kwargs) -> str:
        """Stand-in for ``Agent.run()`` in v3.0.0.

        The real ``Agent.run()`` delegates to ``Nanobot.run()`` which
        uses the upstream ``AgentLoop.process_direct`` under the hood.
        For the AgentService contract under test (record + return), this
        mock is equivalent.
        """
        self.call_count += 1
        sess = self.sessions.setdefault(session_id, {"messages": []})
        sess["messages"].append({"role": "user", "content": content})
        sess["messages"].append({"role": "assistant", "content": self._response})
        self.last_messages = list(sess["messages"])
        return self._response

    async def chat(
        self, content: str, session_id: str = "default", **kwargs
    ):
        """Stand-in for ``Agent.chat()`` — async generator of v2.x events.

        v2.x ``Agent.chat`` is an async generator that yields
        ``{"type": "token" | "done" | ..., "content": ...}`` events.
        AgentService.stream_message consumes this stream.
        """
        self.call_count += 1
        sess = self.sessions.setdefault(session_id, {"messages": []})
        sess["messages"].append({"role": "user", "content": content})
        sess["messages"].append({"role": "assistant", "content": self._response})
        self.last_messages = list(sess["messages"])
        # Yield a single 'done' event carrying the full response.
        # v2.x AgentService's stream_message looks for ``type == "done"``
        # and uses the carried content as final.
        yield {"type": "done", "content": self._response}


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------

def _make_service(tmpdir: str, response: str = "Hello") -> AgentService:
    """Build an ``AgentService`` whose ``_agent`` is a ``MockAgent``."""
    service = AgentService(workspace=tmpdir)
    mock_agent = MockAgent(response=response)
    service._agent = mock_agent  # type: ignore[attr-defined]
    return service, mock_agent


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

class TestAgentServiceSendMessage:
    """``send_message`` returns content + records user/assistant to session."""

    def test_send_message_returns_response_content(self):
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, loop = _make_service(tmp, response="Hi there")
                result = await service.send_message("hello", session_id="s1")
                assert result["content"] == "Hi there"
                assert "message_id" in result
                assert loop.call_count == 1
        asyncio.run(_test())

    def test_send_message_session_has_two_messages(self):
        """After one send, the session should have 2 messages (user + assistant).

        Contract (preserved from v2.x): AgentService delegates
        persistence to the upstream ``AgentLoop``, not to itself. So
        after one round-trip the session holds exactly 2 messages
        — no duplicates from AgentService re-saving.
        """
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, _ = _make_service(tmp, response="Hello")
                await service.send_message("Hi", session_id="test_dedup")
                sess = service._agent.session_manager.get_session("test_dedup")
                assert len(sess.messages) == 2, (
                    f"Expected 2 messages (user + assistant), got {len(sess.messages)}: "
                    f"{[m['content'][:30] for m in sess.messages]}"
                )
                assert sess.messages[0]["role"] == "user"
                assert sess.messages[0]["content"] == "Hi"
                assert sess.messages[1]["role"] == "assistant"
                assert sess.messages[1]["content"] == "Hello"
        asyncio.run(_test())

    def test_send_message_error_no_orphan(self):
        """When the loop raises, no message should be persisted (no orphan)."""
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, mock_agent = _make_service(tmp, response="OK")
                # Replace run with one that always raises
                async def failing_run(content, session_id="default", **kwargs):
                    raise RuntimeError("simulated failure")
                mock_agent.run = failing_run

                result = await service.send_message(
                    "Will fail", session_id="test_error"
                )
                # AgentService catches the exception and returns error result
                assert (
                    "Error" in result.get("content", "")
                    or result.get("error")
                )

                sess = service._agent.session_manager.get_session("test_error")
                assert len(sess.messages) == 0, (
                    f"Expected 0 messages after error, got {len(sess.messages)}"
                )
        asyncio.run(_test())

    def test_send_message_multiple_rounds_correct_count(self):
        """5 rounds → 10 messages (5 user + 5 assistant)."""
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, _ = _make_service(tmp, response="Reply")
                for i in range(5):
                    await service.send_message(f"Q{i}", session_id="multi")
                sess = service._agent.session_manager.get_session("multi")
                assert len(sess.messages) == 10
        asyncio.run(_test())


class TestAgentServiceStreamMessage:
    """``stream_message`` yields v2.x-compatible event dicts."""

    def test_stream_message_emits_done_event(self):
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, _ = _make_service(tmp, response="Stream reply")
                events = []
                async for event in service.stream_message(
                    "Stream test", session_id="s_stream"
                ):
                    events.append(event)
                done = [e for e in events if e.get("type") == "done"]
                assert len(done) == 1
                assert done[0]["content"] == "Stream reply"
        asyncio.run(_test())

    def test_stream_message_session_has_two_messages(self):
        """After streaming, session has 2 messages (user + assistant)."""
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, _ = _make_service(tmp, response="Stream response")
                async for _ in service.stream_message("Hi", session_id="s_stream_2"):
                    pass
                sess = service._agent.session_manager.get_session("s_stream_2")
                assert len(sess.messages) == 2
        asyncio.run(_test())


class TestAgentServiceHistory:
    """``get_history`` returns serialized session messages."""

    def test_get_history_after_send(self):
        async def _test():
            with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
                service, _ = _make_service(tmp, response="Reply")
                await service.send_message("Q1", session_id="hist")
                await service.send_message("Q2", session_id="hist")
                history = service.get_history("hist")
                # 4 messages: Q1(user) + Reply(assistant) + Q2(user) + Reply(assistant)
                assert len(history) == 4
        asyncio.run(_test())

    def test_list_sessions_returns_session_metadata(self):
        """``list_sessions`` returns a list of session dicts (may be empty)."""
        with tempfile.TemporaryDirectory(prefix="test_svc_") as tmp:
            service, _ = _make_service(tmp)
            sessions = service.list_sessions()
            # list_sessions can be a list of dicts or an empty list
            assert isinstance(sessions, list)
