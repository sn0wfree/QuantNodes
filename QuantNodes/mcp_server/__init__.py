# coding=utf-8
"""QuantNodes MCP server.

Exposes the 8 core quant tools as MCP (Model Context Protocol) tools so
they can be invoked by:
- HKUDS/nanobot 0.2.1 (via nanobot_config.json mcpServers entry)
- Any MCP client (Claude Desktop, Cursor, VSCode, ...)
- A second nanobot instance pointing at this server

Run as stdio (default for nanobot)::

    python -m QuantNodes.mcp_server

Run with HTTP transport (for testing / inspector)::

    python -m QuantNodes.mcp_server --transport http --port 8765

Inspect with MCP devtools::

    mcp dev QuantNodes.mcp_server.server:app
"""

from .server import app, mcp, main

__version__ = "3.0.0"

__all__ = ["app", "mcp", "main", "__version__"]
