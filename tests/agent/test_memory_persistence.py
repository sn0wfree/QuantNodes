# coding=utf-8
"""
测试 Memory Persistence 系统 (Phase A-F)

Agent Memory Persistence 设计文档完整测试覆盖
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from QuantNodes.agent.core.memory import (
    Dream,
    DreamConfig,
    DreamStore,
    MemoryStore,
    MemoryManager,
)
from QuantNodes.agent.core.dream import DreamEngine
from QuantNodes.agent.core.autocompact import truncate_history, microcompact
from QuantNodes.agent.session.manager import SessionManager, Session


# ============================================================
# Phase A: 统一 Session 存储
# ============================================================


class TestPhaseASessionUnification:
    """Phase A: 验证 SessionManager 替代 AgentService._sessions"""

    def test_session_manager_get_session_creates_new(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            session = sm.get_session("new-session")
            assert session.session_id == "new-session"
            assert session.messages == []

    def test_session_manager_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            session = sm.get_session("persist-test")
            session.add_message("user", "hello")
            sm.save_session(session)

            # Reload from disk
            sm2 = SessionManager(Path(tmpdir))
            session2 = sm2.get_session("persist-test")
            assert len(session2.messages) == 1
            assert session2.messages[0]["content"] == "hello"

    def test_session_manager_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            session = sm.get_session("to-delete")
            sm.save_session(session)
            assert sm.delete_session("to-delete") is True
            assert sm.delete_session("nonexistent") is False

    def test_session_manager_get_session_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            session = sm.get_session("info-test")
            session.add_message("user", "hello")
            sm.save_session(session)

            info = sm.get_session_info("info-test")
            assert info is not None
            assert info["session_id"] == "info-test"
            assert info["message_count"] == 1
            assert "created_at" in info
            assert "updated_at" in info

    def test_session_manager_get_session_info_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            info = sm.get_session_info("does-not-exist")
            assert info is None

    def test_session_manager_list_sessions_with_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            for name in ["s1", "s2", "s3"]:
                session = sm.get_session(name)
                session.add_message("user", f"msg-{name}")
                sm.save_session(session)

            infos = sm.list_sessions_with_info()
            assert len(infos) == 3
            assert all("session_id" in i for i in infos)
            assert all("message_count" in i for i in infos)

    def test_session_manager_list_sessions_sorted_by_updated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SessionManager(Path(tmpdir))
            for name in ["old", "new"]:
                session = sm.get_session(name)
                session.add_message("user", name)
                sm.save_session(session)

            infos = sm.list_sessions_with_info()
            assert infos[0]["session_id"] == "new"
            assert infos[1]["session_id"] == "old"


# ============================================================
# Phase B: history.jsonl 增强
# ============================================================


class TestPhaseBHistoryEnhancement:
    """Phase B: 验证 history.jsonl 增强"""

    def test_append_history_with_tools_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history(
                {"session_key": "test", "user": "hello", "assistant": "hi"},
                tools_used=["factor", "wiki"],
            )
            history_file = Path(tmpdir) / "memory" / "history.jsonl"
            with open(history_file) as f:
                data = json.loads(f.readline())
            assert data["tools_used"] == ["factor", "wiki"]
            assert "timestamp" in data

    def test_append_history_with_insights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history(
                {"session_key": "test", "user": "q", "assistant": "a"},
                insights=["IC均值0.032"],
            )
            history_file = Path(tmpdir) / "memory" / "history.jsonl"
            with open(history_file) as f:
                data = json.loads(f.readline())
            assert data["insights"] == ["IC均值0.032"]

    def test_get_recent_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            for i in range(5):
                store.append_history(
                    {"session_key": f"s{i}", "user": f"q{i}", "assistant": f"a{i}"}
                )
            recent = store.get_recent_history(limit=3)
            assert len(recent) == 3
            assert recent[0]["user"] == "q2"

    def test_get_recent_history_filter_by_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history({"session_key": "s1", "user": "q1"})
            store.append_history({"session_key": "s2", "user": "q2"})
            store.append_history({"session_key": "s1", "user": "q3"})

            result = store.get_recent_history(session_key="s1")
            assert len(result) == 2
            assert all(r["session_key"] == "s1" for r in result)

    def test_search_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history(
                {"session_key": "s1", "user": "分析IC分布", "assistant": "IC均值0.03"}
            )
            store.append_history(
                {"session_key": "s1", "user": "今天天气", "assistant": "晴天"}
            )

            results = store.search_history("IC")
            assert len(results) == 1
            assert "IC" in results[0]["user"]

    def test_search_history_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history({"session_key": "s1", "user": "hello"})

            results = store.search_history("nonexistent")
            assert len(results) == 0


# ============================================================
# Phase C: MemoryManager (Claude Code 风格)
# ============================================================


class TestPhaseCMemoryManager:
    """Phase C: 验证 MemoryManager"""

    def test_read_index_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            index = mm.read_index()
            assert index == "# Memory Index\n"

    def test_write_and_read_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            content = "# Memory Index\n\n## Factors\n- IC经验"
            mm.write_index(content)
            assert mm.read_index() == content

    def test_read_write_topic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            mm.write_topic("factor", "# Factor Notes\nIC分析经验")
            assert mm.read_topic("factor") == "# Factor Notes\nIC分析经验"

    def test_read_topic_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            assert mm.read_topic("nonexistent") == ""

    def test_list_topics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            mm.write_topic("factor", "content1")
            mm.write_topic("strategy", "content2")
            topics = mm.list_topics()
            assert "factor" in topics
            assert "strategy" in topics

    def test_get_memory_context_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            ctx = mm.get_memory_context()
            assert ctx == ""

    def test_get_memory_context_with_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(Path(tmpdir))
            mm.write_index("# Memory Index\n\n## Factors\n- IC经验")
            ctx = mm.get_memory_context()
            assert "记忆索引" in ctx
            assert "IC经验" in ctx


# ============================================================
# Phase D: Dream 集成
# ============================================================


class TestPhaseDDreamIntegration:
    """Phase D: 验证 Dream 对话分析"""

    def test_should_analyze_conversation_with_keywords(self):
        engine = DreamEngine(DreamStore(Path("/tmp/test_dream")))
        assert engine.should_analyze_conversation("帮我分析IC分布", "IC均值0.03") is True

    def test_should_analyze_conversation_without_keywords(self):
        engine = DreamEngine(DreamStore(Path("/tmp/test_dream")))
        assert engine.should_analyze_conversation("今天天气", "晴天") is False

    @pytest.mark.asyncio
    async def test_analyze_conversation_generates_dream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            engine = DreamEngine(store)

            dream = await engine.analyze_conversation(
                user_message="帮我分析沪深300的IC分布",
                assistant_response="IC均值0.032，ICIR为0.67，因子有效",
            )
            assert dream is not None
            assert dream.type == "conversation_insight"
            assert dream.confidence > 0.6
            assert len(dream.insights) > 0

    @pytest.mark.asyncio
    async def test_analyze_conversation_no_insight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            engine = DreamEngine(store)

            dream = await engine.analyze_conversation(
                user_message="今天天气怎么样",
                assistant_response="今天晴天",
            )
            assert dream is None

    @pytest.mark.asyncio
    async def test_analyze_conversation_user_preference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            engine = DreamEngine(store)

            dream = await engine.analyze_conversation(
                user_message="记住我偏好用QMT回测",
                assistant_response="好的，已记录您的偏好",
            )
            assert dream is not None
            assert any("偏好" in i for i in dream.insights)

    def test_dream_config_new_fields(self):
        config = DreamConfig()
        assert config.min_rounds_before_activate == 5
        assert config.compaction_dream_interval == 5
        assert "IC" in config.analysis_keywords
        assert "因子" in config.analysis_keywords


# ============================================================
# Phase F: Compaction-Dream 集成
# ============================================================


class TestPhaseFCompactionDream:
    """Phase F: 验证截断消息 Dream 分析"""

    def test_truncate_history_returns_tuple(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        kept, dropped = truncate_history(messages, max_messages=20)
        assert len(kept) == 20
        assert len(dropped) == 10

    def test_truncate_history_no_dropping(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        kept, dropped = truncate_history(messages, max_messages=20)
        assert len(kept) == 5
        assert dropped == []

    def test_truncate_history_preserves_order(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        kept, dropped = truncate_history(messages, max_messages=20)
        assert kept[0]["content"] == "msg10"
        assert kept[-1]["content"] == "msg29"
        assert dropped[0]["content"] == "msg0"
        assert dropped[-1]["content"] == "msg9"

    def test_truncate_history_keeps_system(self):
        messages = [{"role": "system", "content": "sys"}]
        messages += [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        kept, dropped = truncate_history(messages, max_messages=20)
        assert kept[0]["role"] == "system"
        assert len(kept) == 21

    def test_dream_config_compaction_interval(self):
        config = DreamConfig()
        assert config.compaction_dream_interval == 5


# ============================================================
# Phase E: AgentLoop 集成
# ============================================================


class TestPhaseEAgentLoopIntegration:
    """Phase E: 验证 AgentLoop 集成所有组件"""

    def test_agent_loop_init_components(self):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus
        from QuantNodes.agent.providers.base import LLMProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()

            class MockProvider(LLMProvider):
                async def chat(self, messages, tools=None, model=None,
                              max_tokens=1024, temperature=0.7, tool_choice=None):
                    from QuantNodes.agent.providers.base import LLMResponse
                    return LLMResponse(content="mock")

            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            assert hasattr(loop, 'session_manager')
            assert hasattr(loop, 'memory')
            assert hasattr(loop, 'memory_manager')
            assert hasattr(loop, 'dream_engine')
            assert hasattr(loop, '_pending_dream_analysis')
            assert hasattr(loop, '_compaction_counter')
            assert loop._compaction_counter == 0
            assert loop._pending_dream_analysis == []

    def test_inject_memory_context(self):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus
        from QuantNodes.agent.providers.base import LLMProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()

            class MockProvider(LLMProvider):
                async def chat(self, messages, tools=None, model=None,
                              max_tokens=1024, temperature=0.7, tool_choice=None):
                    from QuantNodes.agent.providers.base import LLMResponse
                    return LLMResponse(content="mock")

            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            # Write some memory
            loop.memory_manager.write_index("# Memory Index\n\n## Factors\n- IC经验")

            messages = [{"role": "system", "content": "You are helpful."}]
            loop._inject_memory_context(messages, session_key="test")

            assert "IC经验" in messages[0]["content"]

    def test_inject_memory_context_no_system_message(self):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus
        from QuantNodes.agent.providers.base import LLMProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()

            class MockProvider(LLMProvider):
                async def chat(self, messages, tools=None, model=None,
                              max_tokens=1024, temperature=0.7, tool_choice=None):
                    from QuantNodes.agent.providers.base import LLMResponse
                    return LLMResponse(content="mock")

            provider = MockProvider()
            loop = AgentLoop(bus, provider, Path(tmpdir))

            messages = [{"role": "user", "content": "hello"}]
            loop._inject_memory_context(messages)
            assert messages[0]["content"] == "hello"

    def test_chat_simple_with_memory(self):
        async def _test():
            from QuantNodes.agent.core.loop import AgentLoop
            from QuantNodes.agent.bus.queue import MessageBus
            from QuantNodes.agent.providers.base import LLMProvider, LLMResponse

            with tempfile.TemporaryDirectory() as tmpdir:
                bus = MessageBus()

                class MockProvider(LLMProvider):
                    async def chat(self, messages, tools=None, model=None,
                                  max_tokens=1024, temperature=0.7, tool_choice=None):
                        return LLMResponse(content="Mock response")

                provider = MockProvider()
                loop = AgentLoop(bus, provider, Path(tmpdir))

                result = await loop.chat("Hi there", session_id="test_mem")
                assert result == "Mock response"

                session = loop.session_manager.get_session("test_mem")
                assert len(session.messages) == 2

        asyncio.run(_test())


# ============================================================
# DreamStore 集成
# ============================================================


class TestDreamStoreIntegration:
    """验证 DreamStore 与 Memory 的集成"""

    def test_save_and_get_dreams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            dream = Dream(
                id="test-1",
                timestamp="2026-05-10T12:00:00",
                type="factor_insight",
                content="因子分析结果",
                insights=["IC均值0.03"],
                confidence=0.85,
            )
            store.save_dream(dream)
            dreams = store.get_recent_dreams()
            assert len(dreams) == 1
            assert dreams[0].type == "factor_insight"

    def test_get_injection_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            dream = Dream(
                id="test-2",
                timestamp="2026-05-10T12:00:00",
                type="conversation_insight",
                content="对话摘要",
                insights=["洞察1"],
                confidence=0.8,
            )
            store.save_dream(dream)
            config = DreamConfig()
            content = store.get_injection_content(config)
            assert "对话摘要" in content

    def test_get_injection_content_filters_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            dream = Dream(
                id="test-3",
                timestamp="2026-05-10T12:00:00",
                type="low_confidence",
                content="低置信度",
                confidence=0.3,
            )
            store.save_dream(dream)
            config = DreamConfig(min_confidence=0.7)
            content = store.get_injection_content(config)
            assert content == ""

    def test_get_dreams_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DreamStore(Path(tmpdir))
            store.save_dream(Dream(
                id="1", timestamp="2026-05-10T12:00:00",
                type="factor", content="f1", confidence=0.8
            ))
            store.save_dream(Dream(
                id="2", timestamp="2026-05-10T12:00:01",
                type="strategy", content="s1", confidence=0.8
            ))
            store.save_dream(Dream(
                id="3", timestamp="2026-05-10T12:00:02",
                type="factor", content="f2", confidence=0.8
            ))

            factors = store.get_dreams_by_type("factor")
            assert len(factors) == 2
            assert all(d.type == "factor" for d in factors)
