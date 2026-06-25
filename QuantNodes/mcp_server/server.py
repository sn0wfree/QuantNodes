# coding=utf-8
"""FastMCP server exposing QuantNodes quant tools over MCP.

Approach
--------
Each QuantNodes tool is a class with ``name``, ``description``, ``parameters``
(JSON Schema), and ``async execute(**kwargs)``. FastMCP doesn't support
``**kwargs`` function signatures, so we use a single ``call_quant_tool``
dispatcher MCP tool that:
1. Accepts ``tool_name`` + arbitrary ``arguments`` dict (object).
2. Looks up the QuantNodes tool by name.
3. Invokes ``tool.execute(**arguments)`` and returns the result.

This is cleaner than registering 8 separate MCP tools (which would require
dynamic signature generation) and provides a uniform contract.

Run as stdio (default for nanobot)::

    python -m QuantNodes.mcp_server

Run with HTTP transport (for inspector)::

    python -m QuantNodes.mcp_server --transport http --port 8765

Inspect with MCP devtools::

    mcp dev QuantNodes.mcp_server.server:app
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import Field, create_model

from QuantNodes.constants import DEFAULT_HOST, DEFAULT_WEBSOCKET_PORT

logger = logging.getLogger(__name__)


_DEFAULT_WORKSPACE = Path(".agent")

mcp = FastMCP(
    name="quant",
    instructions=(
        "QuantNodes quant toolkit dispatcher. Use `list_quant_tools` to discover "
        "available tools, then `call_quant_tool(tool_name, arguments)` to invoke. "
        "Each tool accepts a JSON Schema-defined arguments object."
    ),
)


def _build_tools_registry(workspace: Path) -> Dict[str, Any]:
    """Instantiate the 8 core quant tools (lazy + cached)."""
    if hasattr(mcp, "_quant_tools_cache") and mcp._quant_tools_cache is not None:
        return mcp._quant_tools_cache

    try:
        from QuantNodes.agent.tools import (
            BacktestTool,
            ConfigBacktestTool,
            FactorTool,
            PipelineTool,
            SandboxTool,
            StrategyTool,
            WikiTool,
        )
        workspace_dep = {"WikiTool": {"wiki_path": str(workspace / "wiki")}}
        factories = [BacktestTool, ConfigBacktestTool, FactorTool, PipelineTool,
                     SandboxTool, StrategyTool, WikiTool]
    except Exception as exc:
        logger.warning("QuantNodes tool stack unavailable: %s", exc)
        mcp._quant_tools_cache = {}
        return mcp._quant_tools_cache

    registry: Dict[str, Any] = {}
    for factory in factories:
        kwargs = workspace_dep.get(factory.__name__, {})
        try:
            tool = factory(**kwargs) if kwargs else factory()
        except Exception as exc:
            logger.warning("Failed to instantiate %s: %s", factory.__name__, exc)
            continue
        registry[tool.name] = tool
    mcp._quant_tools_cache = registry
    return registry


def _safe_field_type(prop: Dict[str, Any]):
    """Map JSON Schema type to a Python type suitable for Pydantic Field."""
    t = prop.get("type", "string")
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(t, Any)


def _build_pydantic_model(name: str, schema: Dict[str, Any]):
    """Build a Pydantic model from a JSON Schema object for a tool's arguments."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: Dict[str, Any] = {}
    for pname, pschema in props.items():
        py_type = _safe_field_type(pschema)
        default = ... if pname in required else pschema.get("default", None)
        description = pschema.get("description", "")
        fields[pname] = (py_type, Field(default=default, description=description))
    if not fields:
        return create_model(name)
    return create_model(name, **fields)


def _register_tool_call_dispatcher(workspace: Path) -> None:
    """Register a dynamic Pydantic-model-based MCP tool for each QuantNodes tool.

    Each tool's arguments are typed via a dynamically generated Pydantic model,
    so FastMCP can produce a proper JSON Schema in the MCP tool listing.
    """
    registry = _build_tools_registry(workspace)
    for tool_name, tool in registry.items():
        schema = tool.parameters if hasattr(tool, "parameters") else {"type": "object", "properties": {}}
        model_name = f"{tool_name.title().replace('_', '')}Args"
        ArgsModel = _build_pydantic_model(model_name, schema)

        async def _call(arguments: Dict[str, Any], _tool=tool) -> Dict[str, Any]:
            try:
                result = await _tool.execute(**arguments)
                if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
                    return {"status": "ok", "result": result}
                return {"status": "ok", "result": str(result)}
            except Exception as exc:
                logger.exception("Tool %s failed", tool_name)
                return {"status": "error", "tool": tool_name, "error": str(exc)}

        _call.__name__ = f"call_{tool_name}"
        _call.__doc__ = (
            f"Call QuantNodes tool '{tool_name}'. {tool.description}"
        )

        decorated = mcp.tool(
            name=f"call_{tool_name}",
            description=_call.__doc__,
            exclude_args=["_tool"],
        )
        decorated(_call)


def _register_list_tools(workspace: Path) -> None:
    """Register a `list_quant_tools` MCP tool that lists available tools + schemas."""

    @mcp.tool
    async def list_quant_tools() -> Dict[str, Any]:
        """List all available QuantNodes tools with their JSON Schema.

        Returns a dict with keys:
        - tools: list of {name, description, parameters, mcp_name}
        - count: number of tools
        """
        registry = _build_tools_registry(workspace)
        tools = []
        for tname, tool in registry.items():
            tools.append({
                "name": tname,
                "description": tool.description,
                "parameters": tool.parameters,
                "mcp_name": f"call_{tname}",
            })
        return {"count": len(tools), "tools": tools}

    list_quant_tools.__doc__ = (
        "List all available QuantNodes tools (with JSON Schema). "
        "Use this to discover what tools are exposed via this MCP server."
    )


def _register_data_query(workspace: Path) -> None:
    """Register a data_query MCP tool — DuckDB query (v0)."""
    from pydantic import Field as _F

    class DataQueryArgs:
        pass

    @mcp.tool
    async def data_query(sql: str = _F(..., description="Read-only SQL query to execute"),
                         source: str = _F("duckdb", description="Data source: duckdb or clickhouse")) -> Dict[str, Any]:
        """Execute a read-only SQL query against the default DuckDB source (v0)."""
        if source != "duckdb":
            return {"status": "error", "error": f"source={source} not yet supported in v0"}
        try:
            import duckdb
            db_path = workspace / "quantnodes.db"
            if not db_path.exists():
                return {"status": "error", "error": f"db not found: {db_path}"}
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                rows = con.execute(sql).fetchall()
                cols = [d[0] for d in con.description] if con.description else []
                return {
                    "status": "ok",
                    "columns": cols,
                    "rows": [list(r) for r in rows[:1000]],
                    "row_count": len(rows),
                }
            finally:
                con.close()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# Eager registration at import time
_register_tool_call_dispatcher(_DEFAULT_WORKSPACE)
_register_list_tools(_DEFAULT_WORKSPACE)
_register_data_query(_DEFAULT_WORKSPACE)


# FastMCP exposes the ASGI app as ``mcp.http_app`` for HTTP transports
app = mcp.http_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QuantNodes MCP server (exposes quant tools to nanobot/Claude/Cursor)"
    )
    parser.add_argument("--transport", choices=["stdio", "http", "sse", "streamable-http"],
                        default="stdio", help="MCP transport (default: stdio)")
    parser.add_argument("--port", type=int, default=DEFAULT_WEBSOCKET_PORT,
                        help="HTTP port (only for http/sse transports)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="HTTP host (only for http/sse transports)")
    parser.add_argument("--workspace", default=".agent",
                        help="QuantNodes workspace path (default: .agent)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    ws = Path(args.workspace).expanduser().resolve()
    if ws != _DEFAULT_WORKSPACE.resolve():
        mcp._quant_tools_cache = None
        _register_tool_call_dispatcher(ws)
        _register_list_tools(ws)
        _register_data_query(ws)

    logger.info("Starting QuantNodes MCP server (transport=%s, workspace=%s)",
                args.transport, ws)
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=not args.verbose)
    elif args.transport in ("http", "streamable-http"):
        mcp.run(transport=args.transport, host=args.host, port=args.port,
                show_banner=not args.verbose)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port,
                show_banner=not args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
