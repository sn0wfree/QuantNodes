# coding=utf-8
"""
测试主循环 (AgentLoop)
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from QuantNodes.agent.core.loop import AgentLoop
from QuantNodes.agent.providers.base import LLMProvider, LLMResponse
from QuantNodes.agent.tools.echo import EchoTool
from QuantNodes.agent.bus.queue import MessageBus


class MockProvider(LLMProvider):
    def __init__(self, response: str = "Mock response"):
        super().__init__()
        self._response = response
        self.call_count = 0

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self._response)


class TestAgentLoop:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()
            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            assert loop.bus == bus
            assert loop.provider == provider
            assert loop._concurrency_gate._value == 1

    def test_register_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()
            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            loop.register_tool(EchoTool())
            assert loop.tool_registry.get("echo") is not None

    def test_get_session_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()
            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            lock1 = loop.get_session_lock("session1")
            lock2 = loop.get_session_lock("session1")
            lock3 = loop.get_session_lock("session2")

            assert lock1 == lock2
            assert lock1 != lock3

    def test_chat_simple(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                bus = MessageBus()
                provider = MockProvider(response="Hello Agent")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                result = await loop.chat("Hi there", session_id="test_session")

                assert result == "Hello Agent"
                assert provider.call_count == 1

        asyncio.run(_test())

    def test_chat_persists_session(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                bus = MessageBus()
                provider = MockProvider(response="Response")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                await loop.chat("Message 1", session_id="session1")

                session = loop.session_manager.get_session("session1")
                assert len(session.messages) == 2
                assert session.messages[0]["content"] == "Message 1"
                assert session.messages[1]["content"] == "Response"

        asyncio.run(_test())

    def test_chat_multiple_messages_same_session(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                bus = MessageBus()
                provider = MockProvider(response="R")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                await loop.chat("M1", session_id="s1")
                await loop.chat("M2", session_id="s1")

                session = loop.session_manager.get_session("s1")
                assert len(session.messages) == 4

        asyncio.run(_test())

    def test_chat_different_sessions(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                bus = MessageBus()
                provider = MockProvider(response="R")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                await loop.chat("M1", session_id="s1")
                await loop.chat("M2", session_id="s2")

                s1 = loop.session_manager.get_session("s1")
                s2 = loop.session_manager.get_session("s2")

                assert len(s1.messages) == 2
                assert len(s2.messages) == 2
                assert s1.messages[0]["content"] == "M1"
                assert s2.messages[0]["content"] == "M2"

        asyncio.run(_test())
