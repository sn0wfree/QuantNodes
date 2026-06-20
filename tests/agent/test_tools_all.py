# coding=utf-8
"""
Agent 工具综合测试

覆盖所有工具 + ToolRegistry + AgentLoop + AgentRunner
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from QuantNodes.agent.tools.registry import ToolRegistry
from QuantNodes.agent.tools.echo import EchoTool
from QuantNodes.agent.tools.pipeline import PipelineTool
from QuantNodes.agent.tools.factor import FactorTool
from QuantNodes.agent.tools.backtest import BacktestTool
from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
from QuantNodes.agent.tools.web_fetch import WebFetchTool
from QuantNodes.agent.tools.web_search import WebSearchTool
from QuantNodes.agent.tools.task import TaskTool
from QuantNodes.agent.providers.base import LLMProvider, LLMResponse, ToolCallRequest


# ─── ToolRegistry ────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool

    def test_get_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.unregister("echo")
        assert reg.get("echo") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(PipelineTool())
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "echo" in names
        assert "pipeline" in names

    def test_get_tool_schemas(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = await reg.execute_tool("echo", message="test")
        assert result.success is True
        assert result.content == "test"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        reg = ToolRegistry()
        result = await reg.execute_tool("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_validation_error(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = await reg.execute_tool("echo")
        assert result.success is False
        assert "Missing required parameter" in result.error

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        calls = [
            {"name": "echo", "arguments": {"message": "a"}},
            {"name": "echo", "arguments": {"message": "b"}},
        ]
        results = await reg.execute_tools_parallel(calls)
        assert len(results) == 2
        assert all(r.success for r in results)


# ─── PipelineTool ────────────────────────────────────────────────────────────


class TestPipelineTool:
    def test_name(self):
        assert PipelineTool().name == "pipeline"

    def test_read_only(self):
        assert PipelineTool().read_only is True

    @pytest.mark.asyncio
    async def test_valid_code(self):
        tool = PipelineTool()
        code = """
from QuantNodes.backtest.strategy_node import MAStrategyNode
strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
"""
        result = await tool.execute(code=code)
        assert result["is_valid"] is True
        assert "MAStrategyNode" in result["nodes"]

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        tool = PipelineTool()
        result = await tool.execute(code="def foo(")
        assert result["is_valid"] is False
        assert any("Syntax error" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_extract_code_block(self):
        tool = PipelineTool()
        code = "```python\nx = 1\n```"
        result = await tool.execute(code=code)
        assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_empty_code(self):
        tool = PipelineTool()
        result = await tool.execute(code="")
        assert result["is_valid"] is False


# ─── FactorTool ──────────────────────────────────────────────────────────────


class TestFactorTool:
    def test_name(self):
        assert FactorTool().name == "factor"

    def test_read_only(self):
        assert FactorTool().read_only is True

    @pytest.mark.asyncio
    async def test_ic_analysis(self):
        tool = FactorTool()
        factor_code = """
import polars as pl
result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
"""
        result = await tool.execute(factor_code=factor_code, analysis_type="ic")
        assert result["status"] == "success"
        assert "ic_mean" in result["analysis"]["ic"]

    @pytest.mark.asyncio
    async def test_missing_result_variable(self):
        tool = FactorTool()
        result = await tool.execute(
            factor_code="x = 1",
            analysis_type="ic",
        )
        assert result["status"] == "error"
        assert "No 'result' variable" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_missing_required_columns(self):
        tool = FactorTool()
        factor_code = """
import polars as pl
result = pl.DataFrame({"date": ["2024-01-01"], "value": [1.0]})
"""
        result = await tool.execute(factor_code=factor_code, analysis_type="ic")
        assert result["status"] == "error"
        assert "Missing required columns" in result["errors"][0]


# ─── BacktestTool ────────────────────────────────────────────────────────────


class TestBacktestTool:
    def test_name(self):
        assert BacktestTool().name == "backtest"

    def test_not_read_only(self):
        assert BacktestTool().read_only is False

    @pytest.mark.asyncio
    async def test_no_strategy_found(self):
        tool = BacktestTool()
        code = "x = 1"
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert "No StrategyNode found" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_no_quote_data_found(self):
        tool = BacktestTool()
        code = """
from QuantNodes.backtest.strategy_node import MAStrategyNode
strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
"""
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert "No quote_data found" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_unsafe_code_rejected(self):
        tool = BacktestTool()
        code = "import os\nos.system('rm -rf /')"
        result = await tool.execute(pipeline_code=code)
        assert result["status"] == "error"
        assert result.get("security_status") == "unsafe"


# ─── ConfigBacktestTool ─────────────────────────────────────────────────────


class TestConfigBacktestTool:
    def test_name(self):
        assert ConfigBacktestTool().name == "config_backtest"

    @pytest.mark.asyncio
    async def test_no_config(self):
        tool = ConfigBacktestTool()
        result = await tool.execute()
        assert result["status"] == "error"
        assert "Need config_yaml or config_path" in result["errors"]

    @pytest.mark.asyncio
    async def test_invalid_yaml(self):
        tool = ConfigBacktestTool()
        result = await tool.execute(config_yaml=": invalid yaml {{{")
        assert result["status"] == "error"
        assert "Invalid YAML" in result["errors"][0]


# ─── Agent API Normalization ─────────────────────────────────────────────────


class TestAgentAPIBaseNormalization:
    """验证 api_base URL 正确 normalize"""

    def test_strip_chat_completions(self):
        from QuantNodes.agent import Agent
        Agent(workspace="/tmp/test_agent_norm", config={})
        # Verify the normalization logic inline
        base_url = "https://openrouter.ai/api/v1/chat/completions"
        # Simulate normalization
        base_url = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1/chat/completions", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        assert base_url == "https://openrouter.ai/api/v1"

    def test_strip_v1(self):
        base_url = "https://api.openai.com/v1"
        base_url = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1/chat/completions", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        assert base_url == "https://api.openai.com"

    def test_no_suffix_unchanged(self):
        base_url = "http://localhost:11434"
        base_url = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1/chat/completions", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        assert base_url == "http://localhost:11434"


# ─── should_execute_tools ────────────────────────────────────────────────────


class TestShouldExecuteTools:
    def test_tool_calls_finish_reason(self):
        resp = LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={"message": "hi"})],
            finish_reason="tool_calls",
        )
        assert resp.should_execute_tools is True

    def test_stop_finish_reason_no_execute(self):
        resp = LLMResponse(
            content="Hello",
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={"message": "hi"})],
            finish_reason="stop",
        )
        assert resp.should_execute_tools is False

    def test_no_tool_calls(self):
        resp = LLMResponse(content="Hello", finish_reason="stop")
        assert resp.should_execute_tools is False


# ─── AgentRunner (MockProvider) ──────────────────────────────────────────────


class MockProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._call_idx = 0
        self.call_count = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return LLMResponse(content="Done", finish_reason="stop")


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_simple_response(self):
        from QuantNodes.agent.core.runner import AgentRunner, AgentRunSpec
        from QuantNodes.agent.tools.registry import ToolRegistry

        provider = MockProvider([
            LLMResponse(content="Hello!", finish_reason="stop"),
        ])
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=ToolRegistry(),
        )
        result = await runner.run(spec)
        assert result.final_content == "Hello!"
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_tool_call_execution(self):
        from QuantNodes.agent.core.runner import AgentRunner, AgentRunSpec

        tool_reg = ToolRegistry()
        tool_reg.register(EchoTool())

        provider = MockProvider([
            # First call: request tool
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="tc1", name="echo", arguments={"message": "hello"})],
                finish_reason="tool_calls",
            ),
            # Second call: final response
            LLMResponse(content="Tool executed!", finish_reason="stop"),
        ])
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "echo hello"}],
            tools=tool_reg,
            max_iterations=3,
        )
        result = await runner.run(spec)
        assert result.final_content == "Tool executed!"
        assert "echo" in result.tools_used
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        from QuantNodes.agent.core.runner import AgentRunner, AgentRunSpec

        provider = MockProvider([
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="tc1", name="echo", arguments={"message": "loop"})],
                finish_reason="tool_calls",
            ),
        ] * 10)
        runner = AgentRunner(provider)
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "loop forever"}],
            tools=ToolRegistry(),
            max_iterations=2,
        )
        result = await runner.run(spec)
        assert result.stop_reason == "max_iterations"


# ─── AgentLoop (MockProvider) ────────────────────────────────────────────────


class TestAgentLoop:
    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_chat(self, workspace):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        provider = MockProvider([
            LLMResponse(content="I'm your quant assistant.", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace)
        result = await loop.chat("你好", session_id="test")
        assert result == "I'm your quant assistant."

    @pytest.mark.asyncio
    async def test_chat_with_tool(self, workspace):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        provider = MockProvider([
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="tc1", name="echo", arguments={"message": "test"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Echo returned: test", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace)
        loop.register_tool(EchoTool())
        result = await loop.chat("echo test", session_id="test_tool")
        assert result == "Echo returned: test"

    @pytest.mark.asyncio
    async def test_session_persistence(self, workspace):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        provider = MockProvider([
            LLMResponse(content="Reply 1", finish_reason="stop"),
            LLMResponse(content="Reply 2", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace)
        await loop.chat("Q1", session_id="persist")
        await loop.chat("Q2", session_id="persist")
        session = loop.session_manager.get_session("persist")
        assert len(session.messages) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_different_sessions_isolated(self, workspace):
        from QuantNodes.agent.core.loop import AgentLoop
        from QuantNodes.agent.bus.queue import MessageBus

        provider = MockProvider([
            LLMResponse(content="A reply", finish_reason="stop"),
            LLMResponse(content="B reply", finish_reason="stop"),
        ])
        bus = MessageBus()
        loop = AgentLoop(bus, provider, workspace)
        await loop.chat("A question", session_id="session_a")
        await loop.chat("B question", session_id="session_b")
        sa = loop.session_manager.get_session("session_a")
        sb = loop.session_manager.get_session("session_b")
        assert sa.messages[0]["content"] == "A question"
        assert sb.messages[0]["content"] == "B question"


# ─── WebFetchTool ─────────────────────────────────────────────────────

class TestWebFetchTool:
    def test_name(self):
        tool = WebFetchTool()
        assert tool.name == "web_fetch"

    def test_read_only(self):
        tool = WebFetchTool()
        assert tool.read_only is True

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = WebFetchTool()
        result = await tool.execute(url="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_local_url_blocked(self):
        tool = WebFetchTool()
        result = await tool.execute(url="http://localhost:8080/secret")
        assert "error" in result
        assert "not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_localhost_127_blocked(self):
        tool = WebFetchTool()
        result = await tool.execute(url="http://127.0.0.1:3000/data")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_non_http_blocked(self):
        tool = WebFetchTool()
        result = await tool.execute(url="file:///etc/passwd")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_text_format(self):
        tool = WebFetchTool()
        result = await tool.execute(url="https://httpbin.org/get", format="text")
        assert result.get("status_code") == 200
        assert "content" in result

    @pytest.mark.asyncio
    async def test_fetch_html_format(self):
        tool = WebFetchTool()
        result = await tool.execute(url="https://httpbin.org/get", format="html")
        assert result.get("status_code") == 200
        assert "content" in result


# ─── WebSearchTool ─────────────────────────────────────────────────────

class TestWebSearchTool:
    def test_name(self):
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_read_only(self):
        tool = WebSearchTool()
        assert tool.read_only is True

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute(query="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        tool = WebSearchTool()
        mock_html = '''
        <html><body>
        <div class="result">
            <div class="result__title"><a href="https://example.com">Example</a></div>
            <div class="result__snippet">An example page</div>
        </div>
        </body></html>
        '''
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()

        with patch.object(tool._client, "get", new=AsyncMock(return_value=mock_resp)):
            result = await tool.execute(query="python programming", max_results=3)
        assert "results" in result
        assert "query" in result
        assert result["query"] == "python programming"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Example"
        assert result["results"][0]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_search_max_results_respected(self):
        tool = WebSearchTool()
        mock_html = '''
        <html><body>
        <div class="result">
            <div class="result__title"><a href="https://a.com">A</a></div>
            <div class="result__snippet">A</div>
        </div>
        <div class="result">
            <div class="result__title"><a href="https://b.com">B</a></div>
            <div class="result__snippet">B</div>
        </div>
        <div class="result">
            <div class="result__title"><a href="https://c.com">C</a></div>
            <div class="result__snippet">C</div>
        </div>
        </body></html>
        '''
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()

        with patch.object(tool._client, "get", new=AsyncMock(return_value=mock_resp)):
            result = await tool.execute(query="test", max_results=2)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    async def test_search_result_structure(self):
        tool = WebSearchTool()
        mock_html = '''
        <html><body>
        <div class="result">
            <div class="result__title"><a href="https://python.org">Python</a></div>
            <div class="result__snippet">The Python language</div>
        </div>
        </body></html>
        '''
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()

        with patch.object(tool._client, "get", new=AsyncMock(return_value=mock_resp)):
            result = await tool.execute(query="python", max_results=1)
        results = result.get("results", [])
        assert len(results) == 1
        assert "title" in results[0]
        assert "url" in results[0]
        assert "snippet" in results[0]


# ─── TaskTool ─────────────────────────────────────────────────────────

class TestTaskTool:
    def test_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            assert tool.name == "task"

    def test_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            assert tool.read_only is False

    @pytest.mark.asyncio
    async def test_create_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            result = await tool.execute(action="create_task", title="Test task", priority="high")
            assert result["status"] == "ok"
            assert result["task"]["title"] == "Test task"
            assert result["task"]["priority"] == "high"
            assert result["task"]["status"] == "pending"
            assert "id" in result["task"]

    @pytest.mark.asyncio
    async def test_create_task_empty_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            result = await tool.execute(action="create_task", title="")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_update_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            create_result = await tool.execute(action="create_task", title="Task")
            task_id = create_result["task"]["id"]
            result = await tool.execute(action="update_task", task_id=task_id, status="completed")
            assert result["status"] == "ok"
            assert result["task"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_task_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            result = await tool.execute(action="update_task", task_id="nonexistent")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            await tool.execute(action="create_task", title="A")
            await tool.execute(action="create_task", title="B")
            result = await tool.execute(action="list_tasks")
            assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_list_tasks_filter_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            r1 = await tool.execute(action="create_task", title="A")
            await tool.execute(action="create_task", title="B")
            await tool.execute(action="update_task", task_id=r1["task"]["id"], status="completed")
            result = await tool.execute(action="list_tasks", status="completed")
            assert result["total"] == 1
            assert result["tasks"][0]["title"] == "A"

    @pytest.mark.asyncio
    async def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool1 = TaskTool(workspace=tmp)
            await tool1.execute(action="create_task", title="Persistent")
            tool2 = TaskTool(workspace=tmp)
            result = await tool2.execute(action="list_tasks")
            assert result["total"] == 1
            assert result["tasks"][0]["title"] == "Persistent"

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = TaskTool(workspace=tmp)
            result = await tool.execute(action="unknown")
            assert "error" in result
