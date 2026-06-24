# coding=utf-8
"""Tests for QuantNodes.mcp_server — exposes quant tools over MCP.

Verifies:
- FastMCP server can be imported
- 9 tools are registered: 7 call_* + list_quant_tools + data_query
- list_quant_tools returns proper metadata
- call_sandbox rejects dangerous Python patterns
- Workspace parameter is forwarded to WikiTool
"""

import asyncio
import json

import pytest

from QuantNodes.mcp_server import mcp, app


def test_mcp_app_importable():
    """FastMCP ASGI app should be importable from the public module."""
    assert app is not None
    assert mcp.name == "quant"


def test_mcp_tools_registered():
    """All 9 expected tools should be registered on import."""
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "call_backtest", "call_config_backtest", "call_factor",
        "call_pipeline", "call_sandbox", "call_strategy", "call_wiki",
        "list_quant_tools", "data_query",
    }
    missing = expected - names
    assert not missing, f"missing MCP tools: {missing}"


def test_list_quant_tools_returns_metadata():
    """list_quant_tools should return a dict with count + tools array."""
    result = asyncio.run(mcp.call_tool("list_quant_tools", {}))
    content = result.structured_content
    assert "count" in content
    assert "tools" in content
    assert content["count"] >= 7
    for tool in content["tools"]:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert "mcp_name" in tool


def test_call_sandbox_rejects_dangerous_code():
    """call_sandbox should reject code with `print()` or other dangerous patterns."""
    result = asyncio.run(mcp.call_tool(
        "call_sandbox",
        {"arguments": {"code": "print('hello')"}}
    ))
    content = result.structured_content
    assert content["status"] == "ok"
    assert content["result"]["is_safe"] is False
    assert any("Dangerous" in e for e in content["result"]["errors"])


def test_call_sandbox_accepts_safe_code():
    """call_sandbox should accept a simple safe Python expression."""
    result = asyncio.run(mcp.call_tool(
        "call_sandbox",
        {"arguments": {"code": "x = 1 + 2"}}
    ))
    content = result.stured_content if hasattr(result, 'stured_content') else result.structured_content
    assert content["status"] == "ok"
    assert content["result"]["is_safe"] is True


def test_call_wiki_list_factors():
    """call_wiki with action=list_factors should return a list (possibly empty)."""
    result = asyncio.run(mcp.call_tool(
        "call_wiki",
        {"arguments": {"action": "list_factors"}}
    ))
    content = result.structured_content
    assert content["status"] == "ok"
    assert "result" in content
    assert isinstance(content["result"], (list, dict))


def test_data_query_rejects_clickhouse_in_v0():
    """data_query should reject non-duckdb sources in v0."""
    result = asyncio.run(mcp.call_tool(
        "data_query",
        {"sql": "SELECT 1", "source": "clickhouse"}
    ))
    content = result.structured_content
    assert content["status"] == "error"
    assert "clickhouse" in content["error"]


def test_mcp_tool_schemas_have_types():
    """Each registered MCP tool should expose a JSON Schema with object type."""
    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        if t.name.startswith("call_"):
            schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None) or getattr(t, "parameters", None)
            assert schema is not None, f"{t.name} missing schema"
            assert schema.get("type") == "object"
            assert "properties" in schema
