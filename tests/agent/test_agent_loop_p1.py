# coding=utf-8
"""
P1 Fixes Tests

验证 AgentLoop 三项 P1 修复：
- P1-1: Dream 洞察注入 system prompt
- P1-2: _pending_dream_analysis per-session 隔离
- P1-3: _process_message 异常时发送错误 OutboundMessage
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


from QuantNodes.agent.providers.base import LLMProvider, LLMResponse
from QuantNodes.agent.core.memory import Dream


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


class FailingProvider(LLMProvider):
    """模拟始终失败的 Provider"""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="", error="Simulated LLM failure")


# ─── P1-1: Dream 注入 System Prompt ────────────────────────────────────


class TestDreamInjection:
    """验证 DreamStore 洞察被注入到 system prompt"""

    def _make_loop(self, tmpdir, provider=None):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        bus = MessageBus()
        provider = provider or MockProvider(response="OK")
        return AgentLoop(bus, provider, Path(tmpdir))

    def test_inject_includes_dream_insights(self):
        """_inject_memory_context 注入了 DreamStore 中的高置信度 Dream"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)

                # 写入一个高置信度 Dream
                dream = Dream(
                    id="test_1",
                    timestamp=datetime.now().isoformat(),
                    type="factor_insight",
                    content="IC均值 0.05，因子有效",
                    insights=["因子 momentum_20d 的 ICIR > 0.5"],
                    confidence=0.9,
                )
                loop.dream_store.save_dream(dream)

                messages = [{"role": "system", "content": "You are a quant agent."}]
                loop._inject_memory_context(messages, "test_session")

                content = messages[0]["content"]
                assert "Dream Insights" in content
                assert "factor_insight" in content
                assert "IC均值 0.05" in content

        asyncio.run(_test())

    def test_inject_no_dream_when_empty(self):
        """DreamStore 为空时不注入"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)

                messages = [{"role": "system", "content": "You are a quant agent."}]
                loop._inject_memory_context(messages, "test_session")

                content = messages[0]["content"]
                assert "Dream Insights" not in content

        asyncio.run(_test())

    def test_inject_dream_respects_confidence_threshold(self):
        """低置信度 Dream 不注入"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)

                # 低置信度 Dream（低于 min_confidence=0.7）
                dream = Dream(
                    id="low_1",
                    timestamp=datetime.now().isoformat(),
                    type="test",
                    content="low confidence insight",
                    confidence=0.3,
                )
                loop.dream_store.save_dream(dream)

                messages = [{"role": "system", "content": "You are a quant agent."}]
                loop._inject_memory_context(messages, "test_session")

                content = messages[0]["content"]
                assert "Dream Insights" not in content

        asyncio.run(_test())


# ─── P1-2: Per-Session Dream Analysis ────────────────────────────────────


class TestPerSessionDreamAnalysis:
    """验证 _pending_dream_analysis 按 session 隔离"""

    def _make_loop(self, tmpdir, provider=None):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        bus = MessageBus()
        provider = provider or MockProvider(response="OK")
        return AgentLoop(bus, provider, Path(tmpdir))

    def test_pending_is_dict_not_list(self):
        """_pending_dream_analysis 是 Dict 而非 List"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)
                assert isinstance(loop._pending_dream_analysis, dict)
                assert isinstance(loop._compaction_counter, dict)

        asyncio.run(_test())

    def test_sessions_isolated_pending(self):
        """不同 session 的 dropped 消息隔离存储"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)

                # 模拟两个 session 各自有 dropped 消息
                loop._pending_dream_analysis["session_a"] = [
                    {"role": "user", "content": "session A message"}
                ]
                loop._pending_dream_analysis["session_b"] = [
                    {"role": "user", "content": "session B message"}
                ]

                pending_a = loop._pending_dream_analysis.get("session_a", [])
                pending_b = loop._pending_dream_analysis.get("session_b", [])

                assert len(pending_a) == 1
                assert len(pending_b) == 1
                assert pending_a[0]["content"] == "session A message"
                assert pending_b[0]["content"] == "session B message"

        asyncio.run(_test())

    def test_compaction_counter_per_session(self):
        """_compaction_counter 按 session 隔离"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                loop = self._make_loop(tmpdir)

                loop._compaction_counter["session_a"] = 3
                loop._compaction_counter["session_b"] = 1

                assert loop._compaction_counter["session_a"] == 3
                assert loop._compaction_counter["session_b"] == 1

        asyncio.run(_test())


# ─── P1-3: Error Recovery ────────────────────────────────────────────────


class TestProcessMessageErrorRecovery:
    """验证 _process_message 异常时发送错误 OutboundMessage"""

    def _make_loop(self, tmpdir, provider=None):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        bus = MessageBus()
        provider = provider or MockProvider(response="OK")
        return AgentLoop(bus, provider, Path(tmpdir))

    def test_error_sends_outbound_message(self):
        """runner.run 抛异常时仍发送 OutboundMessage"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus
                from QuantNodes.agent.bus.events import InboundMessage

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                # 让 runner.run 抛异常
                original_run = loop.runner.run

                async def failing_run(spec):
                    raise RuntimeError("Simulated runner failure")

                loop.runner.run = failing_run

                msg = InboundMessage(
                    channel="test",
                    sender_id="user1",
                    chat_id="chat1",
                    content="Hello",
                )

                await loop._process_message(msg)

                # 应该收到了错误 OutboundMessage
                outbound = await bus.consume_outbound()
                assert "错误" in outbound.content
                assert "Simulated runner failure" in outbound.content

        asyncio.run(_test())

    def test_error_no_session_write(self):
        """异常时不写入 session"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus
                from QuantNodes.agent.bus.events import InboundMessage

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                async def failing_run(spec):
                    raise RuntimeError("Simulated runner failure")

                loop.runner.run = failing_run

                msg = InboundMessage(
                    channel="test",
                    sender_id="user1",
                    chat_id="chat1",
                    content="Hello",
                )

                await loop._process_message(msg)

                # Session 应该是空的（没有 user/assistant 消息写入）
                session = loop.session_manager.get_session("test:chat1")
                assert len(session.messages) == 0

        asyncio.run(_test())

    def test_error_no_history_write(self):
        """异常时不写入 history.jsonl"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus
                from QuantNodes.agent.bus.events import InboundMessage

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                async def failing_run(spec):
                    raise RuntimeError("Simulated runner failure")

                loop.runner.run = failing_run

                msg = InboundMessage(
                    channel="test",
                    sender_id="user1",
                    chat_id="chat1",
                    content="Hello",
                )

                await loop._process_message(msg)

                # history.jsonl 不应该有新增记录
                history = loop.memory.get_recent_history(limit=100)
                assert len(history) == 0

        asyncio.run(_test())

    def test_error_preserves_existing_session(self):
        """异常不影响已有的 session 数据"""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                from QuantNodes.agent.core.loop import AgentLoop
                from QuantNodes.agent.bus.queue import MessageBus
                from QuantNodes.agent.bus.events import InboundMessage

                bus = MessageBus()
                provider = MockProvider(response="OK")
                loop = AgentLoop(bus, provider, Path(tmpdir))

                # 先成功处理一条消息
                msg1 = InboundMessage(
                    channel="test",
                    sender_id="user1",
                    chat_id="chat1",
                    content="First message",
                )
                await loop._process_message(msg1)

                session = loop.session_manager.get_session("test:chat1")
                assert len(session.messages) == 2  # user + assistant

                # 再让 runner 失败
                async def failing_run(spec):
                    raise RuntimeError("Simulated runner failure")

                loop.runner.run = failing_run

                msg2 = InboundMessage(
                    channel="test",
                    sender_id="user1",
                    chat_id="chat1",
                    content="Second message",
                )
                await loop._process_message(msg2)

                # 之前的消息应该还在
                session = loop.session_manager.get_session("test:chat1")
                assert len(session.messages) == 2  # 仍然是 2 条（第一条的）
                assert session.messages[0]["content"] == "First message"

        asyncio.run(_test())
