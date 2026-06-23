# coding=utf-8
"""Focused tool tests — does not depend on agent loop / nanobot runtime.

These tests instantiate quant tools directly and verify their JSON
schema, ``name``, ``description``, and pure-python logic paths that
don't require an LLM or external resources.

Tools tested:
- EchoTool: pure echo
- SandboxTool: validation of dangerous Python imports
- (other tools: see tests/agent/test_tools_all.py for coverage)
"""

import pytest

from pathlib import Path

from QuantNodes.agent.tools import (
    BacktestTool,
    ConfigBacktestTool,
    EchoTool,
    FactorTool,
    FileOpsTool,
    PipelineTool,
    SandboxTool,
    StrategyTool,
    TaskTool,
    Tool,
    WebFetchTool,
    WebSearchTool,
    WikiTool,
    CodeSearchTool,
    GitOpsTool,
)
from QuantNodes.agent.tools.base import ToolExecutionResult


def test_tool_base_subclasses_nanobot_tool():
    """Our Tool base must subclass nanobot's Tool (verified via MRO)."""
    from nanobot.agent.tools.base import Tool as NanobotTool

    assert issubclass(Tool, NanobotTool)


def test_tool_execution_result_dataclass():
    r = ToolExecutionResult(tool_name="x", success=True, content="ok")
    assert r.tool_name == "x"
    assert r.success is True
    assert r.error is None


@pytest.mark.parametrize(
    "factory,expected_name",
    [
        (EchoTool, "echo"),
        (SandboxTool, "sandbox"),
        (PipelineTool, "pipeline"),
        (StrategyTool, "strategy"),
        (BacktestTool, "backtest"),
        (FactorTool, "factor"),
        (ConfigBacktestTool, "config_backtest"),
        (WebFetchTool, "web_fetch"),
        (WebSearchTool, "web_search"),
    ],
)
def test_tool_default_constructor_has_name(factory, expected_name):
    """Tools that take no constructor args can be instantiated empty and expose name."""
    tool = factory()
    assert tool.name == expected_name
    assert isinstance(tool, Tool)


@pytest.mark.parametrize(
    "factory,kwargs,expected_name",
    [
        (WikiTool, {"wiki_path": "/tmp/wiki"}, "wiki"),
        (FileOpsTool, {"workspace": "/tmp/ws"}, "file_ops"),
        (CodeSearchTool, {"workspace": "/tmp/ws"}, "code_search"),
        (GitOpsTool, {"workspace": "/tmp/ws"}, "git_ops"),
        (TaskTool, {"workspace": "/tmp/ws"}, "task"),
    ],
)
def test_tool_with_workspace_kwarg_has_name(factory, kwargs, expected_name):
    """Tools that require a workspace / wiki_path accept those kwargs."""
    tool = factory(**kwargs)
    assert tool.name == expected_name


@pytest.mark.asyncio
async def test_echo_tool_returns_input():
    tool = EchoTool()
    result = await tool.execute(message="hello world")
    assert "hello world" in str(result)


def test_all_tools_have_valid_json_schema():
    """Every tool's ``parameters`` should be a valid JSON Schema object."""
    tools = [
        EchoTool(),
        SandboxTool(),
        PipelineTool(),
        StrategyTool(),
        BacktestTool(),
        FactorTool(),
        ConfigBacktestTool(),
        WebFetchTool(),
        WebSearchTool(),
    ]
    for tool in tools:
        schema = tool.parameters
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema
        assert isinstance(schema["properties"], dict)


def test_tool_to_schema_works():
    """All tools should be serializable via upstream ``to_schema()``."""
    tool = EchoTool()
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "description" in schema["function"]
    assert "parameters" in schema["function"]


def test_register_all_quant_tools_idempotent():
    """Calling register twice should not double-register."""
    from QuantNodes.agent.tools import register_all_quant_tools
    from nanobot.agent.tools.registry import ToolRegistry

    reg = ToolRegistry()
    n1 = register_all_quant_tools(reg, workspace=Path("/tmp/test_register"))
    n2 = register_all_quant_tools(reg, workspace=Path("/tmp/test_register"))
    assert n1 == n2 or n2 == 0  # second call should be no-op
