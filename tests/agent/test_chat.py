# coding=utf-8
"""
Agent Chat 测试用例
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from QuantNodes.agent.core.loop import AgentLoop
from QuantNodes.agent.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from QuantNodes.agent.bus.queue import MessageBus
from QuantNodes.agent.tools.echo import EchoTool


class MockProvider(LLMProvider):
    """模拟 LLM Provider"""
    
    def __init__(self, response: str = "Mock response", tool_calls: List[ToolCallRequest] = None):
        super().__init__()
        self._response = response
        self._tool_calls = tool_calls or []
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
        return LLMResponse(
            content=self._response,
            tool_calls=self._tool_calls
        )


class TestAgentChat:
    """Agent Chat 测试类"""

    @pytest.fixture
    def workspace(self):
        """创建临时工作目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def bus(self):
        """创建消息总线"""
        return MessageBus()

    @pytest.fixture
    def provider(self):
        """创建模拟 Provider"""
        return MockProvider(response="Hello! I'm your quant assistant.")

    @pytest.fixture
    def agent_loop(self, bus, provider, workspace):
        """创建 AgentLoop 实例"""
        loop = AgentLoop(bus, provider, workspace)
        loop.register_tool(EchoTool())
        return loop

    @pytest.mark.asyncio
    async def test_simple_chat(self, agent_loop, provider):
        """测试：简单对话"""
        result = await agent_loop.chat(
            message="你好",
            session_id="test_session"
        )
        
        assert result == "Hello! I'm your quant assistant."
        assert provider.call_count == 1
        assert provider.last_messages is not None

    @pytest.mark.asyncio
    async def test_chat_with_history(self, agent_loop, provider):
        """测试：带历史记录的多轮对话"""
        # 第一轮
        await agent_loop.chat("什么是均线策略?", session_id="session1")
        
        # 第二轮（带历史）
        result = await agent_loop.chat("请给我一个具体例子", session_id="session1")
        
        assert provider.call_count == 2
        # 验证历史消息被传递
        assert len(provider.last_messages) >= 2

    @pytest.mark.asyncio
    async def test_different_sessions_isolated(self, agent_loop, provider):
        """测试：不同会话隔离"""
        await agent_loop.chat("策略A", session_id="session_a")
        await agent_loop.chat("策略B", session_id="session_b")
        
        # 验证两个会话独立
        session_a = agent_loop.session_manager.get_session("session_a")
        session_b = agent_loop.session_manager.get_session("session_b")
        
        assert len(session_a.messages) == 2
        assert len(session_b.messages) == 2
        assert session_a.messages[0]["content"] == "策略A"
        assert session_b.messages[0]["content"] == "策略B"

    @pytest.mark.asyncio
    async def test_session_persistence(self, agent_loop, provider, workspace):
        """测试：会话持久化"""
        session_id = "persist_test"
        
        # 第一次对话
        await agent_loop.chat("测试问题", session_id=session_id)
        
        # 检查会话文件是否保存
        session_file = workspace / "sessions" / f"{session_id}.json"
        assert session_file.exists()

    @pytest.mark.asyncio
    async def test_concurrent_chat(self, agent_loop, provider):
        """测试：并发对话"""
        tasks = [
            agent_loop.chat(f"问题{i}", session_id=f"session_{i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r == "Hello! I'm your quant assistant." for r in results)


class TestAgentChatIntegration:
    """Agent Chat 集成测试"""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        """测试：完整对话流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = MessageBus()
            provider = MockProvider(
                response="这是一个双均线策略的示例代码..."
            )
            loop = AgentLoop(bus, provider, Path(tmpdir))
            
            # 注册工具
            from QuantNodes.agent.tools.echo import EchoTool
            loop.register_tool(EchoTool())
            
            # 执行对话
            result = await loop.chat(
                message="帮我生成一个双均线交叉策略",
                session_id="strategy_gen"
            )
            
            assert result is not None
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])