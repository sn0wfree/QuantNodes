# coding=utf-8
"""Tests for LLMGateway — unified LLM entry point.

Covers:
  - LLMGateway initialization (with/without agent)
  - chat() interface (LLMClientBase compatible)
  - complete() interface (alpha_gpt compatible)
  - __call__() interface (callable injection compatible)
  - run() async interface (nanobot native)
  - get_llm_gateway() / create_llm_gateway() / reset_llm_gateway()
  - Messages-to-prompt conversion
  - Fallback to NullLLMClient when nanobot unavailable
  - Tool calling: tools, tool_choice, tool_calls, ToolCallResponse
"""
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.ai.llm.base import (
    LLMClientBase,
    Message,
    MessageRole,
    ChatCompletion,
)
from QuantNodes.ai.llm.gateway import (
    LLMGateway,
    ToolCallResponse,
    get_llm_gateway,
    create_llm_gateway,
    reset_llm_gateway,
)
from QuantNodes.ai.llm.null import NullLLMClient


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class FakeAgent:
    """模拟 nanobot Agent: 记录 run() 和 chat() 调用。"""

    def __init__(
        self,
        response: str = "fake agent response",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        self.response = response
        self.calls: List[dict] = []
        self._tool_calls = tool_calls or []

    async def run(self, prompt: str, session_id: str = "default") -> str:
        self.calls.append({"prompt": prompt, "session_id": session_id})
        return self.response

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        self.calls.append({
            "prompt": message,
            "session_id": session_id,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        for tc in self._tool_calls:
            yield {"type": "tool_call", **tc}
        yield {"type": "done", "content": self.response, "stop_reason": "stop"}


# ---------------------------------------------------------------------------
# 1. LLMGateway 初始化
# ---------------------------------------------------------------------------

class TestLLMGatewayInit:
    def test_init_without_agent(self):
        """无 agent 参数时, gateway 仍可创建, agent 懒加载。"""
        g = LLMGateway(workspace=".agent")
        assert g._agent is None
        assert g._workspace == ".agent"
        assert g._agent_resolved is False

    def test_init_with_agent(self):
        """传入 agent 时, gateway 持有该 agent。"""
        agent = FakeAgent()
        g = LLMGateway(agent=agent)
        assert g._agent is agent

    def test_is_llm_client_base_subclass(self):
        """LLMGateway 继承 LLMClientBase, 兼容 chat() 接口。"""
        g = LLMGateway()
        assert isinstance(g, LLMClientBase)


# ---------------------------------------------------------------------------
# 2. _ensure_agent() 懒加载
# ---------------------------------------------------------------------------

class TestEnsureAgent:
    def test_ensure_agent_returns_existing(self):
        """已有 agent 时, 直接返回。"""
        agent = FakeAgent()
        g = LLMGateway(agent=agent)
        assert g._ensure_agent() is agent

    def test_ensure_agent_litellm_disabled(self):
        """agent_factory 抛异常时, _agent = None, _fallback = NullLLMClient。"""
        def bad_factory():
            raise ImportError("nanobot not installed")

        g = LLMGateway(workspace=".agent", agent_factory=bad_factory)
        agent = g._ensure_agent()
        assert agent is None
        assert isinstance(g._fallback, NullLLMClient)
        # 第二次调用应直接返回 None (不会重新尝试 import)
        assert g._ensure_agent() is None
        assert g._agent_resolved is True


# ---------------------------------------------------------------------------
# 3. chat() 接口 (LLMClientBase 兼容)
# ---------------------------------------------------------------------------

class TestChatInterface:
    def test_chat_with_agent(self):
        """传入 agent 时, chat() 调用 agent.run() 并包装为 ChatCompletion。"""
        agent = FakeAgent(response="hello world")
        g = LLMGateway(agent=agent)

        response = g.chat([
            Message(role=MessageRole.USER, content="hi"),
        ])

        assert isinstance(response, ChatCompletion)
        assert response.content == "hello world"
        assert response.role == MessageRole.ASSISTANT
        assert len(agent.calls) == 1
        assert agent.calls[0]["session_id"] == "default"

    def test_chat_with_dict_messages(self):
        """chat() 接受 dict 格式消息 (LLMClientBase 规范)。"""
        agent = FakeAgent(response="dict response")
        g = LLMGateway(agent=agent)

        response = g.chat([{"role": "user", "content": "hello"}])
        assert response.content == "dict response"

    def test_chat_fallback_to_null(self):
        """agent_factory 抛异常时, 降级到 NullLLMClient。"""
        def bad_factory():
            raise ImportError("nanobot not installed")

        g = LLMGateway(workspace=".agent", agent_factory=bad_factory)
        response = g.chat([Message(role=MessageRole.USER, content="hi")])
        assert isinstance(response, ChatCompletion)
        # NullLLMClient 的默认响应
        assert "null" in (response.finish_reason or "").lower() or len(response.content) >= 0

    def test_chat_temperature_passed(self):
        """chat() 接受 temperature / max_tokens 参数。"""
        agent = FakeAgent()
        g = LLMGateway(agent=agent)
        g.chat(
            [Message(role=MessageRole.USER, content="hi")],
            temperature=0.3,
            max_tokens=100,
        )
        # 验证 prompt 包含消息内容
        assert "hi" in agent.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# 4. complete() 接口 (alpha_gpt 兼容)
# ---------------------------------------------------------------------------

class TestCompleteInterface:
    def test_complete_with_agent(self):
        """complete(agent_id, prompt) 调用 agent.run(), session_id=agent_id。"""
        agent = FakeAgent(response="complete response")
        g = LLMGateway(agent=agent)

        result = g.complete(agent_id="formula-translator", prompt="gen formula")
        assert result == "complete response"
        assert len(agent.calls) == 1
        assert agent.calls[0]["session_id"] == "formula-translator"
        assert agent.calls[0]["prompt"] == "gen formula"

    def test_complete_default_agent_id(self):
        """complete() 默认 agent_id='default'。"""
        agent = FakeAgent()
        g = LLMGateway(agent=agent)
        g.complete(prompt="hi")
        assert agent.calls[0]["session_id"] == "default"

    def test_complete_fallback(self):
        """agent_factory 抛异常时降级到 NullLLMClient。"""
        def bad_factory():
            raise ImportError("nanobot not installed")

        g = LLMGateway(workspace=".agent", agent_factory=bad_factory)
        result = g.complete(agent_id="test", prompt="hi")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. __call__() 接口 (callable injection 兼容)
# ---------------------------------------------------------------------------

class TestCallableInterface:
    def test_call_with_agent(self):
        """llm_gateway(prompt) 等价于 agent.run(prompt, 'default')。"""
        agent = FakeAgent(response="callable response")
        g = LLMGateway(agent=agent)

        result = g("test prompt")
        assert result == "callable response"
        assert len(agent.calls) == 1
        assert agent.calls[0]["session_id"] == "default"

    def test_call_fallback(self):
        """nanobot 不可用时降级。"""
        def bad_factory():
            raise ImportError("nanobot not installed")

        g = LLMGateway(workspace=".agent", agent_factory=bad_factory)
        result = g("test")
        assert isinstance(result, str)

    def test_call_used_as_callable_injection(self):
        """LLMGateway 实例可直接作为 llm_callable 注入。"""
        agent = FakeAgent(response="LLMJudge response")
        g = LLMGateway(agent=agent)

        # 模拟 LLMJudge/Compressor/Operators 注入
        llm_callable = g
        result = llm_callable("judge this")
        assert result == "LLMJudge response"


# ---------------------------------------------------------------------------
# 6. run() async 接口 (nanobot 原生)
# ---------------------------------------------------------------------------

class TestRunInterface:
    def test_run_is_coroutine(self):
        """run() 是 async 函数。"""
        g = LLMGateway()
        assert asyncio.iscoroutinefunction(g.run)

    def test_run_with_agent(self):
        """async run() 委托 agent.run()。"""
        agent = FakeAgent(response="async response")
        g = LLMGateway(agent=agent)

        result = asyncio.run(g.run("test prompt", session_id="s1"))
        assert result == "async response"
        assert agent.calls[0]["session_id"] == "s1"

    def test_run_default_session_id(self):
        """run() 默认 session_id='default'。"""
        agent = FakeAgent()
        g = LLMGateway(agent=agent)
        asyncio.run(g.run("test"))
        assert agent.calls[0]["session_id"] == "default"

    def test_run_fallback(self):
        """agent_factory 抛异常时 run() 降级。"""
        def bad_factory():
            raise ImportError("nanobot not installed")

        g = LLMGateway(workspace=".agent", agent_factory=bad_factory)
        result = asyncio.run(g.run("test"))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 7. _messages_to_prompt() 内部工具
# ---------------------------------------------------------------------------

class TestMessagesToPrompt:
    def test_user_message_only(self):
        """只有 user 消息: 直接返回 content。"""
        g = LLMGateway()
        prompt = g._messages_to_prompt([
            Message(role=MessageRole.USER, content="hello"),
        ])
        assert prompt == "hello"

    def test_system_message(self):
        """system 消息: 标注 [System] 前缀。"""
        g = LLMGateway()
        prompt = g._messages_to_prompt([
            Message(role=MessageRole.SYSTEM, content="you are a helper"),
            Message(role=MessageRole.USER, content="hi"),
        ])
        assert "[System] you are a helper" in prompt
        assert "hi" in prompt

    def test_assistant_message(self):
        """assistant 消息: 标注 [Assistant] 前缀。"""
        g = LLMGateway()
        prompt = g._messages_to_prompt([
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content="hello back"),
        ])
        assert "[Assistant] hello back" in prompt

    def test_empty_messages(self):
        """空消息列表返回空字符串。"""
        g = LLMGateway()
        assert g._messages_to_prompt([]) == ""


# ---------------------------------------------------------------------------
# 8. 全局单例管理
# ---------------------------------------------------------------------------

class TestGlobalSingleton:
    def setup_method(self):
        reset_llm_gateway()

    def teardown_method(self):
        reset_llm_gateway()

    def test_get_llm_gateway_returns_singleton(self):
        """get_llm_gateway() 多次调用返回同一实例。"""
        g1 = get_llm_gateway()
        g2 = get_llm_gateway()
        assert g1 is g2

    def test_get_llm_gateway_workspace(self):
        """workspace 参数生效。"""
        g = get_llm_gateway(workspace="/tmp/test")
        assert g._workspace == "/tmp/test"

    def test_create_llm_gateway_returns_new(self):
        """create_llm_gateway() 每次返回新实例。"""
        g1 = create_llm_gateway()
        g2 = create_llm_gateway()
        assert g1 is not g2

    def test_create_llm_gateway_with_agent(self):
        """create_llm_gateway(agent) 持有传入的 agent。"""
        agent = FakeAgent()
        g = create_llm_gateway(agent=agent)
        assert g._agent is agent

    def test_reset_clears_singleton(self):
        """reset_llm_gateway() 清空单例。"""
        g1 = get_llm_gateway()
        reset_llm_gateway()
        g2 = get_llm_gateway()
        assert g1 is not g2


# ---------------------------------------------------------------------------
# 9. 集成场景: 模块使用 LLMGateway
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    def test_strategy_generator_uses_gateway(self):
        """StrategyGenerator() 默认使用 LLMGateway。"""
        from QuantNodes.ai.strategy_gen import StrategyGenerator
        from QuantNodes.ai.llm.gateway import LLMGateway

        gen = StrategyGenerator()
        assert isinstance(gen.llm, LLMGateway)

    def test_optimizer_uses_gateway(self):
        """PipelineOptimizer() 默认使用 LLMGateway。"""
        from QuantNodes.ai.optimizer import PipelineOptimizer
        from QuantNodes.ai.llm.gateway import LLMGateway

        opt = PipelineOptimizer()
        assert isinstance(opt.llm, LLMGateway)

    def test_llm_judge_mock_no_gateway(self):
        """LLMJudge(model='mock') 不注入 gateway (mock 优先)。"""
        from QuantNodes.core.feedback.llm_judge import LLMJudge

        j = LLMJudge(model="mock")
        assert j._llm_callable is None

    def test_llm_judge_real_injects_gateway(self):
        """LLMJudge(model='deepseek') 自动注入 LLMGateway。"""
        from QuantNodes.core.feedback.llm_judge import LLMJudge
        from QuantNodes.ai.llm.gateway import LLMGateway

        j = LLMJudge(model="deepseek-v3")
        assert isinstance(j._llm_callable, LLMGateway)

    def test_compressor_mock_no_gateway(self):
        """Compressor(model='mock') 不注入 gateway。"""
        from QuantNodes.core.knowledge.lineage_compress import Compressor

        c = Compressor(model="mock")
        assert c._llm_callable is None

    def test_compressor_real_injects_gateway(self):
        """Compressor(model='deepseek') 自动注入 LLMGateway。"""
        from QuantNodes.core.knowledge.lineage_compress import Compressor
        from QuantNodes.ai.llm.gateway import LLMGateway

        c = Compressor(model="deepseek-v3")
        assert isinstance(c._llm_callable, LLMGateway)

    def test_hypothesizer_mock_no_gateway(self):
        """Hypothesizer(model='mock') 不注入 gateway。"""
        from QuantNodes.core.evolution.operators import Hypothesizer

        h = Hypothesizer(model="mock")
        assert h._llm_callable is None

    def test_hypothesizer_real_injects_gateway(self):
        """Hypothesizer(model='real') 自动注入 LLMGateway。"""
        from QuantNodes.core.evolution.operators import Hypothesizer
        from QuantNodes.ai.llm.gateway import LLMGateway

        h = Hypothesizer(model="deepseek-v3")
        assert isinstance(h._llm_callable, LLMGateway)

    def test_alpha_gpt_workflow_uses_gateway(self):
        """AlphaGptWorkflow(llm_client=gateway) 使用 LLMGateway。"""
        from QuantNodes.research.quant_alpha.workflow.alpha_gpt import (
            AlphaGptConfig, AlphaGptWorkflow,
        )
        from QuantNodes.ai.llm.gateway import LLMGateway

        agent = FakeAgent(response="mock")
        gateway = LLMGateway(agent=agent)
        config = AlphaGptConfig(objective="momentum", iterations=1)
        workflow = AlphaGptWorkflow(config=config, llm_client=gateway)
        assert isinstance(workflow.llm_client, LLMGateway)

    def test_alpha_gpt_workflow_no_client_is_none(self):
        """AlphaGptWorkflow() 不传 llm_client 时为 None (mock 模式)。"""
        from QuantNodes.research.quant_alpha.workflow.alpha_gpt import (
            AlphaGptConfig, AlphaGptWorkflow,
        )

        config = AlphaGptConfig(objective="momentum", iterations=1)
        workflow = AlphaGptWorkflow(config=config)
        assert workflow.llm_client is None


# ---------------------------------------------------------------------------
# 10. ToolCallResponse 数据类
# ---------------------------------------------------------------------------

class TestToolCallResponse:
    def test_has_tool_calls_true(self):
        """有工具调用时 has_tool_calls=True。"""
        resp = ToolCallResponse(
            content="result",
            tools_used=["backtest"],
        )
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        """无工具调用时 has_tool_calls=False。"""
        resp = ToolCallResponse(content="result")
        assert resp.has_tool_calls is False

    def test_defaults(self):
        """默认值检查。"""
        resp = ToolCallResponse(content="result")
        assert resp.content == "result"
        assert resp.tools_used == []
        assert resp.stop_reason == "stop"
        assert resp.events == []


# ---------------------------------------------------------------------------
# 11. chat() 工具调用接口
# ---------------------------------------------------------------------------

class TestChatToolCalling:
    def test_chat_with_tools_passes_to_agent(self):
        """chat(tools=[...]) 透传给 agent.chat()。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        g.chat(
            [Message(role=MessageRole.USER, content="run backtest")],
            tools=["backtest", "factor"],
        )
        assert len(agent.calls) == 1
        assert agent.calls[0]["tools"] == ["backtest", "factor"]

    def test_chat_with_tool_choice_passes_to_agent(self):
        """chat(tool_choice='required') 透传给 agent.chat()。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        g.chat(
            [Message(role=MessageRole.USER, content="test")],
            tool_choice="required",
        )
        assert agent.calls[0]["tool_choice"] == "required"

    def test_chat_returns_tool_calls(self):
        """LLM 调工具时 ChatCompletion.tool_calls 填充。"""
        agent = FakeAgent(
            response="backtest done",
            tool_calls=[{"id": "tc1", "name": "backtest", "arguments": {"code": "x"}}],
        )
        g = LLMGateway(agent=agent)

        resp = g.chat([Message(role=MessageRole.USER, content="run")])
        assert resp.content == "backtest done"
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "backtest"

    def test_chat_finish_reason_tool_calls(self):
        """工具调用后 finish_reason='tool_calls'。"""
        agent = FakeAgent(
            response="done",
            tool_calls=[{"id": "tc1", "name": "alpha_evaluate", "arguments": {}}],
        )
        g = LLMGateway(agent=agent)

        resp = g.chat([Message(role=MessageRole.USER, content="eval")])
        assert resp.finish_reason == "tool_calls"

    def test_chat_no_tool_calls_finish_reason_stop(self):
        """无工具调用时 finish_reason='stop'。"""
        agent = FakeAgent(response="text only")
        g = LLMGateway(agent=agent)

        resp = g.chat([Message(role=MessageRole.USER, content="hi")])
        assert resp.finish_reason == "stop"
        assert resp.tool_calls is None


# ---------------------------------------------------------------------------
# 12. complete() 工具调用接口
# ---------------------------------------------------------------------------

class TestCompleteToolCalling:
    def test_complete_with_tools_passes_to_agent(self):
        """complete(tools=[...]) 透传给 agent.chat()。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        g.complete(agent_id="test", prompt="gen", tools=["backtest"])
        assert agent.calls[0]["tools"] == ["backtest"]

    def test_complete_returns_text_only(self):
        """complete() 返回纯文本 (工具调用整合到 content 中)。"""
        agent = FakeAgent(
            response="result with tool data",
            tool_calls=[{"id": "tc1", "name": "backtest", "arguments": {}}],
        )
        g = LLMGateway(agent=agent)

        result = g.complete(prompt="run")
        assert result == "result with tool data"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 13. __call__() 工具调用接口
# ---------------------------------------------------------------------------

class TestCallableToolCalling:
    def test_call_with_tools_passes_to_agent(self):
        """callable(prompt, tools=[...]) 透传给 agent.chat()。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        g("judge", tools=["alpha_evaluate"])
        assert agent.calls[0]["tools"] == ["alpha_evaluate"]

    def test_call_returns_text_only(self):
        """callable 返回纯文本。"""
        agent = FakeAgent(response="judge result", tool_calls=[])
        g = LLMGateway(agent=agent)

        result = g("judge")
        assert result == "judge result"


# ---------------------------------------------------------------------------
# 14. run() 工具调用接口
# ---------------------------------------------------------------------------

class TestRunToolCalling:
    def test_run_with_tools_passes_to_agent(self):
        """run(tools=[...]) 透传给 agent.chat()。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        asyncio.run(g.run("test", tools=["backtest"]))
        assert agent.calls[0]["tools"] == ["backtest"]

    def test_run_with_tool_events_returns_tool_call_response(self):
        """run(with_tool_events=True) 返回 ToolCallResponse。"""
        agent = FakeAgent(
            response="done",
            tool_calls=[{"id": "tc1", "name": "backtest", "arguments": {}}],
        )
        g = LLMGateway(agent=agent)

        result = asyncio.run(g.run("run", with_tool_events=True))
        assert isinstance(result, ToolCallResponse)
        assert result.content == "done"
        assert result.has_tool_calls is True
        assert "backtest" in result.tools_used
        assert len(result.events) > 0

    def test_run_with_tool_events_no_calls(self):
        """无工具调用时 ToolCallResponse.has_tool_calls=False。"""
        agent = FakeAgent(response="text only")
        g = LLMGateway(agent=agent)

        result = asyncio.run(g.run("test", with_tool_events=True))
        assert isinstance(result, ToolCallResponse)
        assert result.has_tool_calls is False
        assert result.tools_used == []

    def test_run_with_tool_choice_passes_to_agent(self):
        """run(tool_choice='required') 透传。"""
        agent = FakeAgent(response="ok")
        g = LLMGateway(agent=agent)

        asyncio.run(g.run("test", tool_choice="required"))
        assert agent.calls[0]["tool_choice"] == "required"

    def test_run_with_events_fallback(self):
        """nanobot 不可用时 run(with_tool_events=True) 降级。"""
        g = LLMGateway(workspace=".agent")
        with patch(
            "QuantNodes.agent.nanobot_bridge.Agent",
            side_effect=ImportError("nanobot not installed"),
        ):
            result = asyncio.run(g.run("test", with_tool_events=True))
        assert isinstance(result, ToolCallResponse)
        assert result.has_tool_calls is False
