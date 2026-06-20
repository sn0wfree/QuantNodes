# coding=utf-8
"""
测试 AgentService 消息双重保存修复

验证 AgentService 不再自行写入 Session 消息，
所有消息持久化由 AgentLoop 统一负责。
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List


from QuantNodes.agent.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """模拟 LLM Provider"""

    def __init__(self, response: str = "Mock response"):
        super().__init__()
        self._response = response
        self.call_count = 0
        self.last_messages = None

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
        self.last_messages = messages
        return LLMResponse(content=self._response)


class MockFailingProvider(LLMProvider):
    """模拟始终失败的 LLM Provider"""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="", error="Simulated failure")


class TestAgentServiceNoDoubleSave:
    """验证 AgentService 不再双重保存消息"""

    def _make_service(self, tmpdir, provider=None):
        from api.services.agent_service import AgentService
        service = AgentService(workspace=str(tmpdir))
        if provider:
            from QuantNodes.agent import Agent
            from QuantNodes.agent.bus.queue import MessageBus

            bus = MessageBus()
            agent = Agent.__new__(Agent)
            agent._provider = provider
            agent._config = {}

            from QuantNodes.agent.core.loop import AgentLoop
            loop = AgentLoop(bus, provider, Path(tmpdir))
            agent._loop = loop
            agent._agent = agent

            service._agent = agent
        return service

    def test_send_message_session_has_two_messages(self):
        """send_message 后 session 应有 2 条消息（非 4 条）"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="Hello")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                result = await service.send_message("Hi", session_id="test_dedup")
                assert result["content"] == "Hello"

                session = loop.session_manager.get_session("test_dedup")
                assert len(session.messages) == 2, (
                    f"Expected 2 messages, got {len(session.messages)}: "
                    f"{[m['content'][:30] for m in session.messages]}"
                )
                assert session.messages[0]["role"] == "user"
                assert session.messages[0]["content"] == "Hi"
                assert session.messages[1]["role"] == "assistant"
                assert session.messages[1]["content"] == "Hello"

        asyncio.run(_test())

    def test_send_message_llm_receives_no_duplicate_user(self):
        """LLM 收到的 messages 中 user 消息不重复"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="Response")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                await service.send_message("Test query", session_id="test_ctx")

                messages = provider.last_messages
                user_msgs = [m for m in messages if m.get("role") == "user"]
                assert len(user_msgs) == 1, (
                    f"Expected 1 user message in LLM context, got {len(user_msgs)}"
                )
                assert user_msgs[0]["content"] == "Test query"

        asyncio.run(_test())

    def test_send_message_error_no_orphan(self):
        """agent.run() 异常时 session 不产生孤儿 user 消息"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent


                async def failing_chat(message, session_id="default"):
                    raise RuntimeError("Simulated agent failure")

                loop.chat = failing_chat

                result = await service.send_message(
                    "Will fail", session_id="test_error"
                )
                assert "Error" in result["content"]

                session = loop.session_manager.get_session("test_error")
                assert len(session.messages) == 0, (
                    f"Expected 0 messages after error, got {len(session.messages)}"
                )

        asyncio.run(_test())

    def test_send_message_multiple_rounds_correct_count(self):
        """多轮对话后消息数 = 轮数 × 2"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="Reply")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                for i in range(5):
                    await service.send_message(f"Q{i}", session_id="test_multi")

                session = loop.session_manager.get_session("test_multi")
                assert len(session.messages) == 10, (
                    f"Expected 10 messages after 5 rounds, got {len(session.messages)}"
                )

        asyncio.run(_test())

    def test_stream_message_session_has_two_messages(self):
        """stream_message 后 session 应有 2 条消息"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="Stream response")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                events = []
                async for event in service.stream_message(
                    "Stream test", session_id="test_stream"
                ):
                    events.append(event)

                done_events = [e for e in events if e["type"] == "done"]
                assert len(done_events) == 1

                session = loop.session_manager.get_session("test_stream")
                assert len(session.messages) == 2, (
                    f"Expected 2 messages after stream, got {len(session.messages)}"
                )

        asyncio.run(_test())

    def test_stream_message_error_no_orphan(self):
        """stream_message 异常时不产生孤儿消息"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                async def failing_chat_stream(message, session_id="default"):
                    raise RuntimeError("Simulated stream failure")
                    yield  # pragma: no cover

                loop.chat_stream = failing_chat_stream

                events = []
                async for event in service.stream_message(
                    "Will fail", session_id="test_stream_error"
                ):
                    events.append(event)

                error_events = [e for e in events if e["type"] == "error"]
                assert len(error_events) == 1

                session = loop.session_manager.get_session("test_stream_error")
                assert len(session.messages) == 0, (
                    f"Expected 0 messages after stream error, got {len(session.messages)}"
                )

        asyncio.run(_test())

    def test_get_history_after_send(self):
        """send_message 后 get_history 返回正确消息数"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus

                bus = MessageBus()
                provider = MockProvider(response="Reply")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                from api.services.agent_service import AgentService
                service = AgentService(workspace=str(tmpdir))

                from QuantNodes.agent import Agent
                agent = Agent.__new__(Agent)
                agent._loop = loop
                service._agent = agent

                await service.send_message("Q1", session_id="test_hist")
                await service.send_message("Q2", session_id="test_hist")

                history = service.get_history("test_hist")
                assert len(history) == 4, (
                    f"Expected 4 messages in history after 2 rounds, got {len(history)}"
                )

        asyncio.run(_test())
