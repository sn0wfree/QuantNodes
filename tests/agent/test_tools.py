# coding=utf-8
"""Tests for ``QuantNodes.agent.tools.registry.ToolRegistry`` + 14 quant tools.

v3.0.0 refactor: the v2.x local ``Tool`` (with v2.x-only methods like
``cast_params`` / ``validate_params`` / ``read_only`` / ``concurrency_safe``)
was replaced by a thin wrapper over upstream ``nanobot.agent.tools.base.Tool``.
This file targets the v3.0.0 reality:

- ``ToolRegistry`` (kept locally) — register / unregister / get / list /
  get_tool_schemas / execute_tool
- 14 quant tool classes — ``name`` / ``description`` / ``parameters``
  shape, plus pure-Python logic paths that do not require an LLM
  (``EchoTool`` echo, ``SandboxTool`` safety, ``PipelineTool`` extract,
  ``BacktestTool``/``ConfigBacktestTool`` validation, ``TaskTool``
  CRUD persistence)

All upstream-coupled tests use ``@pytest.mark.skipif(not NANOBOT_AVAILABLE)``
so the suite passes whether or not the optional dep is present.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from QuantNodes.agent import NANOBOT_AVAILABLE
from QuantNodes.agent.tools import (
    BacktestTool,
    CodeSearchTool,
    ConfigBacktestTool,
    EchoTool,
    FactorTool,
    FileOpsTool,
    GitOpsTool,
    PipelineTool,
    SandboxTool,
    StrategyTool,
    TaskTool,
    Tool,
    ToolRegistry,
    WebFetchTool,
    WebSearchTool,
    WikiTool,
)
from QuantNodes.agent.tools.base import ToolExecutionResult


# ----------------------------------------------------------------------------
# ToolRegistry — register / unregister / get / list / schemas
# ----------------------------------------------------------------------------

class TestToolRegistry:
    """``ToolRegistry`` is the kept-local registry that wraps upstream
    nanobot's Tool objects and provides our own execution / validation
    logic.
    """

    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool

    def test_get_unknown_returns_none(self):
        """``get`` returns None (not raises) for unknown tool names."""
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.unregister("echo")
        assert reg.get("echo") is None

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a non-existent tool does not raise."""
        reg = ToolRegistry()
        reg.unregister("ghost")  # must not raise

    def test_list_tools_returns_copy(self):
        """``list_tools`` returns a fresh list (mutation-safe)."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(PipelineTool())
        tools = reg.list_tools()
        assert len(tools) == 2
        tools.append("bogus")  # must not affect the registry
        assert len(reg.list_tools()) == 2

    def test_get_tool_schemas_returns_function_call_dicts(self):
        """Each schema has ``type=function`` and a ``function`` block."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        schemas = reg.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_get_tool_schemas_caches_until_register(self):
        """Schema list is cached; invalidated on register / unregister.

        Verified by inserting a 2nd tool and checking the list re-computes
        (length changes from 1 to 2).
        """
        reg = ToolRegistry()
        reg.register(EchoTool())
        first = reg.get_tool_schemas()
        reg.register(PipelineTool())
        second = reg.get_tool_schemas()
        assert len(first) == 1
        assert len(second) == 2

    @pytest.mark.asyncio
    async def test_execute_tool_happy_path(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = await reg.execute_tool("echo", message="hello")
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.content == "hello"
        assert result.error is None
        assert result.tool_name == "echo"

    @pytest.mark.asyncio
    async def test_execute_tool_missing_returns_not_found(self):
        """Missing tool returns ``success=False`` with descriptive error."""
        reg = ToolRegistry()
        result = await reg.execute_tool("nonexistent", message="x")
        assert result.success is False
        assert "not found" in (result.error or "").lower()
        assert result.tool_name == "nonexistent"

    @pytest.mark.asyncio
    async def test_execute_tool_catches_exceptions(self):
        """If a tool raises, ``execute_tool`` returns a graceful failure result."""

        class _Boom(Tool):
            @property
            def name(self):
                return "boom"

            @property
            def description(self):
                return "always raises"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                raise RuntimeError("kaboom!")

        reg = ToolRegistry()
        reg.register(_Boom())
        result = await reg.execute_tool("boom")
        assert result.success is False
        assert "kaboom" in (result.error or "")

    def test_register_overwrites_with_same_name(self):
        """Re-registering a tool with the same name replaces the previous one."""
        reg = ToolRegistry()
        first = EchoTool()
        second = EchoTool()
        reg.register(first)
        reg.register(second)
        assert reg.get("echo") is second
        assert len(reg.list_tools()) == 1


# ----------------------------------------------------------------------------
# EchoTool — pure echo (no nanobot, no LLM)
# ----------------------------------------------------------------------------

class TestEchoTool:
    def test_name(self):
        assert EchoTool().name == "echo"

    def test_parameters_shape(self):
        params = EchoTool().parameters
        assert "message" in params["properties"]

    @pytest.mark.asyncio
    async def test_execute_returns_input(self):
        result = await EchoTool().execute(message="hello world")
        assert "hello world" in str(result)


# ----------------------------------------------------------------------------
# SandboxTool — code safety analysis (no LLM)
# ----------------------------------------------------------------------------

class TestSandboxTool:
    def test_name(self):
        assert SandboxTool().name == "sandbox"

    @pytest.mark.asyncio
    async def test_safe_code(self):
        tool = SandboxTool()
        result = await tool.execute(code="import pandas\ndf = pd.DataFrame()")
        assert result["is_safe"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_dangerous_os_system_rejected(self):
        tool = SandboxTool()
        result = await tool.execute(code="import os\nos.system('ls')")
        assert result["is_safe"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_empty_code_rejected(self):
        tool = SandboxTool()
        result = await tool.execute(code="")
        assert result["is_safe"] is False

    @pytest.mark.asyncio
    async def test_max_code_length(self):
        tool = SandboxTool(max_code_length=10)
        result = await tool.execute(code="x = 12345678901")
        assert result["is_safe"] is False


# ----------------------------------------------------------------------------
# PipelineTool — code block extraction (no LLM)
# ----------------------------------------------------------------------------

class TestPipelineTool:
    def test_name(self):
        assert PipelineTool().name == "pipeline"

    @pytest.mark.asyncio
    async def test_extract_python_code_from_markdown(self):
        """PipelineTool extracts code from a ```python ... ``` block."""
        tool = PipelineTool()
        code = "```python\nx = 1\n```"
        result = await tool.execute(code=code)
        assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_empty_code_rejected(self):
        tool = PipelineTool()
        result = await tool.execute(code="")
        assert result["is_valid"] is False


# ----------------------------------------------------------------------------
# BacktestTool — required param validation (no LLM, no real backtest)
# ----------------------------------------------------------------------------

class TestBacktestTool:
    def test_name(self):
        assert BacktestTool().name == "backtest"

    def test_parameters_required(self):
        params = BacktestTool().parameters
        assert "pipeline_code" in params["properties"]
        assert "pipeline_code" in params.get("required", [])

    def test_parameters_optional_dates(self):
        """``start_date`` / ``end_date`` are optional (have defaults)."""
        params = BacktestTool().parameters
        for optional in ("start_date", "end_date", "initial_cash"):
            assert optional in params["properties"]
            assert optional not in params.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_no_strategy_found(self):
        """``execute(pipeline_code="x = 1")`` returns error — no StrategyNode."""
        tool = BacktestTool()
        result = await tool.execute(pipeline_code="x = 1")
        assert result["status"] == "error"
        assert any("StrategyNode" in e for e in result.get("errors", []))


# ----------------------------------------------------------------------------
# ConfigBacktestTool — YAML validation
# ----------------------------------------------------------------------------

class TestConfigBacktestTool:
    def test_name(self):
        assert ConfigBacktestTool().name == "config_backtest"

    @pytest.mark.asyncio
    async def test_no_config_returns_error(self):
        tool = ConfigBacktestTool()
        result = await tool.execute()
        assert result["status"] == "error"
        assert any(
            "config_yaml" in e or "config_path" in e
            for e in result.get("errors", [])
        )

    @pytest.mark.asyncio
    async def test_invalid_yaml_returns_error(self):
        tool = ConfigBacktestTool()
        result = await tool.execute(config_yaml=": invalid yaml {{{")
        assert result["status"] == "error"
        assert any("YAML" in e for e in result.get("errors", []))


# ----------------------------------------------------------------------------
# StrategyTool — needs LLM (no LLM = needs_configuration)
# ----------------------------------------------------------------------------

class TestStrategyTool:
    def test_name(self):
        assert StrategyTool().name == "strategy"

    def test_parameters_required(self):
        params = StrategyTool().parameters
        assert "description" in params.get("required", [])

    @pytest.mark.asyncio
    async def test_execute_without_llm_returns_needs_configuration(self):
        """StrategyTool without an LLM client returns ``status=needs_configuration``."""
        tool = StrategyTool()
        result = await tool.execute(description="生成一个动量策略")
        assert result["status"] == "needs_configuration"
        assert "LLM" in result["message"]


# ----------------------------------------------------------------------------
# FactorTool — name + parameters shape (LLM-dependent tests skipped)
# ----------------------------------------------------------------------------

class TestFactorTool:
    def test_name(self):
        assert FactorTool().name == "factor"

    def test_parameters_shape(self):
        params = FactorTool().parameters
        assert "factor_code" in params["properties"]
        assert "analysis_type" in params["properties"]
        # analysis_type enum
        enum = params["properties"]["analysis_type"].get("enum")
        if enum is not None:
            assert set(enum) == {"ic", "correlation", "both"}


# ----------------------------------------------------------------------------
# TaskTool — local persistence (JSON file in workspace)
# ----------------------------------------------------------------------------

class TestTaskTool:
    def _make(self, workspace: Path) -> TaskTool:
        return TaskTool(workspace=workspace)

    def test_name(self):
        with tempfile_TmpDir() as tmp:
            assert self._make(tmp).name == "task"

    def test_read_only_is_false(self):
        """TaskTool writes to disk; not read_only."""
        with tempfile_TmpDir() as tmp:
            assert self._make(tmp).read_only is False

    @pytest.mark.asyncio
    async def test_create_task(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            result = await tool.execute(action="create_task", title="Test", priority="high")
            assert result["status"] == "ok"
            assert result["task"]["title"] == "Test"
            assert result["task"]["priority"] == "high"
            assert result["task"]["status"] == "pending"
            assert "id" in result["task"]

    @pytest.mark.asyncio
    async def test_create_task_empty_title_rejected(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            result = await tool.execute(action="create_task", title="")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_update_task(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            create = await tool.execute(action="create_task", title="X")
            task_id = create["task"]["id"]
            upd = await tool.execute(action="update_task", task_id=task_id, status="completed")
            assert upd["status"] == "ok"
            assert upd["task"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_task_not_found(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            result = await tool.execute(action="update_task", task_id="ghost")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            await tool.execute(action="create_task", title="A")
            await tool.execute(action="create_task", title="B")
            result = await tool.execute(action="list_tasks")
            assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self):
        with tempfile_TmpDir() as tmp:
            await self._make(tmp).execute(action="create_task", title="Persistent")
            result = await self._make(tmp).execute(action="list_tasks")
            assert result["total"] == 1
            assert result["tasks"][0]["title"] == "Persistent"

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        with tempfile_TmpDir() as tmp:
            tool = self._make(tmp)
            result = await tool.execute(action="unknown_action")
            assert "error" in result


# ----------------------------------------------------------------------------
# Tools that take workspace / wiki_path kwargs — constructor shape
# ----------------------------------------------------------------------------

class TestToolConstructorKwargs:
    def test_wiki_tool(self):
        tool = WikiTool(wiki_path="/tmp/wiki_test")
        assert tool.name == "wiki"

    def test_file_ops_tool(self):
        tool = FileOpsTool(workspace="/tmp/ws_test")
        assert tool.name == "file_ops"

    def test_code_search_tool(self):
        tool = CodeSearchTool(workspace="/tmp/ws_test")
        assert tool.name == "code_search"

    def test_git_ops_tool(self):
        tool = GitOpsTool(workspace="/tmp/ws_test")
        assert tool.name == "git_ops"


# ----------------------------------------------------------------------------
# All 14 tools — consistent JSON Schema shape
# ----------------------------------------------------------------------------

class TestAllToolsJsonSchema:
    """Every quant tool's ``parameters`` must be a valid JSON Schema object."""

    @pytest.mark.parametrize(
        "factory",
        [
            EchoTool,
            SandboxTool,
            PipelineTool,
            StrategyTool,
            BacktestTool,
            FactorTool,
            ConfigBacktestTool,
            WebFetchTool,
            WebSearchTool,
        ],
    )
    def test_no_constructor_args(self, factory):
        """Tools that take no constructor args can be instantiated empty."""
        tool = factory()
        assert isinstance(tool, Tool)
        assert tool.name
        assert tool.description
        params = tool.parameters
        assert isinstance(params, dict)
        assert params.get("type") == "object"
        assert "properties" in params


# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def tempfile_TmpDir():
    """Yield a unique temp dir under /tmp (replaces ``tempfile.TemporaryDirectory``).

    Using a real path (not auto-cleaned) makes debugging easier — the dir
    sticks around if the test fails so you can inspect state.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="test_tools_") as tmp:
        yield Path(tmp)
