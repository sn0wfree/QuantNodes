# coding=utf-8
"""Agent facade wrapping HKUDS nanobot 0.2.1 (Path A: direct upstream consumption).

Usage (backward-compatible v2.x signature):

    from QuantNodes.agent import Agent
    agent = Agent(workspace=".agent", config={...})
    result = await agent.run("hello", session_id="default")

Under the hood, Agent constructs a ``Nanobot`` from
``.agent/nanobot_config.json`` (generated from ``.env`` by
``config_mapper.py``), injects all 15 quant tools into its ToolRegistry,
and exposes a thin compatibility layer for the streaming event protocol
that the API layer (api/services/agent_service.py) expects.

Reference: docs/13-Agent架构设计.md (v3.0.0) and docs/14-上游nanobot升级指南.md.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from nanobot import Nanobot
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

from .config_mapper import build_nanobot_config, write_nanobot_config
from .tools import register_all_quant_tools

logger = logging.getLogger(__name__)


class Agent:
    """Thin wrapper around HKUDS nanobot's ``Nanobot`` programmatic facade.

    See nanobot_bridge.py module docstring for the full design rationale.
    """

    DEFAULT_WORKSPACE = ".agent"

    def __init__(
        self,
        workspace: str = DEFAULT_WORKSPACE,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        settings = build_nanobot_config(self.workspace, config)
        self.config_path = write_nanobot_config(self.workspace, settings)

        self._bot: Nanobot = Nanobot.from_config(self.config_path, workspace=self.workspace)
        self._loop = self._bot._loop

        quant_count = register_all_quant_tools(self._loop.tools, workspace=self.workspace)
        logger.info(
            "QuantNodes Agent ready (workspace=%s, quant_tools=%d, "
            "upstream_tools=%d)",
            self.workspace,
            quant_count,
            len(self._loop.tools._tools) - quant_count,
        )

    @property
    def loop(self):
        """Backward-compatible access to underlying AgentLoop."""
        return self._loop

    async def run(self, prompt: str, session_id: str = "default") -> str:
        """Single-turn run returning the assistant's final text content."""
        result = await self._bot.run(prompt, session_key=session_id)
        return result.content or ""

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat events using the v2.x event protocol.

        Emits dicts with shape::

            {"type": "token", "content": str}
            {"type": "tool_call", "id": str, "name": str, "arguments": dict}
            {"type": "tool_result", "id": str, "name": str, "content": Any, "success": bool}
            {"type": "done", "content": str, "tools_used": list[str], "stop_reason": str}
            {"type": "error", "content": str}
        """
        from nanobot.agent.hook import SDKCaptureHook

        capture = SDKCaptureHook()
        prev = self._loop._extra_hooks or []
        self._loop._extra_hooks = [capture, *prev]
        try:
            async for event in self._stream_via_loop(message, session_id, model, max_tokens, mode):
                yield event
        finally:
            self._loop._extra_hooks = prev

    async def _stream_via_loop(
        self,
        message: str,
        session_id: str,
        model: Optional[str],
        max_tokens: Optional[int],
        mode: Optional[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Drive the upstream AgentLoop and re-emit events in v2.x protocol."""
        from nanobot.agent.runner import AgentRunSpec

        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": message}],
            tools=self._loop.tool_registry,
            model=model or getattr(self._loop, "model", None),
            max_tokens=max_tokens or getattr(self._loop, "max_tokens", 102400),
            max_iterations=getattr(self._loop, "max_iterations", 5),
        )

        tools_used: List[str] = []
        final_content = ""
        try:
            async for event in self._loop.run_stream(spec):
                etype = event.get("type")
                if etype == "token":
                    yield {"type": "token", "content": event.get("content", "")}
                elif etype == "tool_call":
                    name = event.get("name") or event.get("tool", "")
                    if name:
                        tools_used.append(name)
                    yield {
                        "type": "tool_call",
                        "id": event.get("id", ""),
                        "name": name,
                        "arguments": event.get("arguments", {}),
                    }
                elif etype == "tool_result":
                    yield {
                        "type": "tool_result",
                        "id": event.get("id", ""),
                        "name": event.get("name", ""),
                        "content": event.get("content"),
                        "success": event.get("success", True),
                    }
                elif etype == "done":
                    final_content = event.get("content", "") or final_content
                    yield {
                        "type": "done",
                        "content": final_content,
                        "tools_used": list(dict.fromkeys(tools_used)),
                        "stop_reason": event.get("stop_reason", "stop"),
                    }
                elif etype == "error":
                    yield {"type": "error", "content": event.get("content", "")}
                elif etype == "message":
                    delta = event.get("content")
                    if delta:
                        final_content += delta
        except AttributeError:
            result = await self._loop.process_direct(message, session_key=session_id)
            yield {
                "type": "done",
                "content": getattr(result, "content", "") or "",
                "tools_used": tools_used,
                "stop_reason": getattr(result, "stop_reason", "stop") or "stop",
            }
        except Exception as exc:
            logger.exception("Agent chat failed")
            yield {"type": "error", "content": str(exc)}


__all__ = ["Agent"]
