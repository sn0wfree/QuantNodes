"""MCP server for Agent access via stdio transport.

Exposes 6 tools (Agent invokes):
    1. audit_get_lesson: load full lesson markdown
    2. audit_list_lessons: list lessons with filters
    3. audit_get_code_context: AST-based code context
    4. audit_search_lessons: keyword search
    5. audit_static_precheck: Engine A quick precheck
    6. audit_submit_finding: Agent submits finding

The server uses stdio transport. Each tool returns structured JSON for
Agent consumption.

Note: This server does NOT call any LLM. The Agent (caller) does semantic
judgment and submits findings via audit_submit_finding.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from quantnodes_strategy_audit.core.lesson import LessonLoader
from quantnodes_strategy_audit.engines.context_engine import ContextEngine
from quantnodes_strategy_audit.engines.static_engine import StaticEngine
from quantnodes_strategy_audit.tools import (
    get_code_context_tool,
    get_lesson_tool,
    list_lessons_tool,
    search_lessons_tool,
    static_precheck_tool,
    submit_finding_tool,
)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    Server = None
    stdio_server = None
    Tool = None
    TextContent = None


MCP_INSTRUCTIONS = """quantnodes-strategy-audit (skill)

Context provider for quantitative strategy audit. NOT an LLM caller.

Tools (Agent calls these):
  - audit_get_lesson(lesson_id): Load full lesson document
  - audit_list_lessons(category, severity, auto_checkable): List lessons
  - audit_get_code_context(file, focus_lines, depth): AST code context
  - audit_search_lessons(query, top_k): Keyword search
  - audit_static_precheck(file, lesson_ids): Engine A quick scan
  - audit_submit_finding(finding_data): Agent submits finding

Workflow:
  1. Call audit_list_lessons to find relevant lessons
  2. Call audit_static_precheck to find suspicious locations
  3. Call audit_get_lesson for full check_prompt
  4. Call audit_get_code_context for AST context
  5. Agent does semantic judgment, then audit_submit_finding
"""


async def serve() -> None:
    """Run the MCP server on stdio."""
    if Server is None:
        print(
            "ERROR: mcp package not installed. Install with: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = Server("quantnodes-strategy-audit")

    # Initialize engines
    pkg_dir = Path(__file__).parent.parent
    lessons_dir = pkg_dir / "lessons"
    rules_path = pkg_dir.parent / "rules" / "simple_rules.yaml"

    loader = LessonLoader(builtin_dir=lessons_dir)
    static = StaticEngine(rules_path=rules_path)
    context_engine = ContextEngine(
        lesson_loader=loader,
        static_engine=static,
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="audit_get_lesson",
                description=(
                    "Load complete lesson document (id, title, severity, "
                    "check_prompt, content_markdown). NO file_path exposed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "lesson_id": {"type": "string", "description": "L-NNN ID"},
                    },
                    "required": ["lesson_id"],
                },
            ),
            Tool(
                name="audit_list_lessons",
                description=(
                    "List lessons with optional filters by category/severity/auto_checkable."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        },
                        "auto_checkable": {
                            "type": "string",
                            "enum": ["static", "agent", "partial", "manual"],
                        },
                    },
                },
            ),
            Tool(
                name="audit_get_code_context",
                description=(
                    "Extract AST-based code context (imports, enclosing_function, "
                    "variables, surrounding lines)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "focus_lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "depth": {"type": "integer", "default": 2},
                    },
                    "required": ["file", "focus_lines"],
                },
            ),
            Tool(
                name="audit_search_lessons",
                description="Search lessons by keyword relevance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="audit_static_precheck",
                description=(
                    "Run Engine A static precheck (fast regex-based scan). "
                    "Returns violations grouped by lesson."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "lesson_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional filter by L-NNN",
                        },
                    },
                    "required": ["file"],
                },
            ),
            Tool(
                name="audit_submit_finding",
                description=(
                    "Agent submits a finding from semantic judgment. "
                    "Stored in audit log for later report generation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "lesson_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["VIOLATED", "OK", "UNCLEAR"],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        },
                        "evidence": {"type": "object"},
                        "fix_suggestion": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["file", "lesson_id", "status"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "audit_get_lesson":
                result = get_lesson_tool(context_engine, arguments["lesson_id"])
            elif name == "audit_list_lessons":
                result = list_lessons_tool(
                    context_engine,
                    category=arguments.get("category"),
                    severity=arguments.get("severity"),
                    auto_checkable=arguments.get("auto_checkable"),
                )
            elif name == "audit_get_code_context":
                result = get_code_context_tool(
                    context_engine,
                    file=arguments["file"],
                    focus_lines=arguments["focus_lines"],
                    depth=arguments.get("depth", 2),
                )
            elif name == "audit_search_lessons":
                result = search_lessons_tool(
                    context_engine,
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                )
            elif name == "audit_static_precheck":
                result = static_precheck_tool(
                    context_engine,
                    file=arguments["file"],
                    lesson_ids=arguments.get("lesson_ids"),
                )
            elif name == "audit_submit_finding":
                result = submit_finding_tool(context_engine, arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            result = {"error": str(e), "tool": name}

        return [
            TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))
        ]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for `quantnodes-audit serve-mcp`."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
