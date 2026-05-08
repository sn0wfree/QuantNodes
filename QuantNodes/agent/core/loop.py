# coding=utf-8
"""
主循环 (AgentLoop)

消息总线驱动，处理并发会话
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

from ..bus.events import InboundMessage, OutboundMessage
from ..bus.queue import MessageBus
from ..session.manager import SessionManager
from ..providers.base import LLMProvider
from ..tools.registry import ToolRegistry
from .context import ContextBuilder
from .runner import AgentRunner, AgentRunSpec
from .hook import AgentHook, CompositeHook
from .memory import MemoryStore
from .autocompact import truncate_history


class AgentLoop:
    """Agent主消息循环"""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path | str,
        model: str | None = None,
        max_iterations: int = 5,
        session_manager: SessionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        hook: AgentHook | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = Path(workspace)
        self.model = model
        self.max_iterations = max_iterations
        self.hook = hook or CompositeHook()

        templates_dir = self.workspace.parent / "templates" / "agent"
        if not templates_dir.exists():
            templates_dir = Path(__file__).parent.parent / "templates" / "agent"

        self.context_builder = ContextBuilder(templates_dir)
        self.session_manager = session_manager or SessionManager(self.workspace)
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory = MemoryStore(self.workspace)
        self.runner = AgentRunner(provider, hook=self.hook)

        self._concurrency_gate = asyncio.Semaphore(1)
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._pending_queues: Dict[str, asyncio.Queue] = {}
        self._active_tasks: Dict[str, list[asyncio.Task]] = {}
        self._running = False

    def register_tool(self, tool: Any) -> None:
        """注册工具"""
        self.tool_registry.register(tool)

    def get_session_lock(self, session_key: str) -> asyncio.Lock:
        """获取会话级锁"""
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    async def run(self) -> None:
        """主循环 - 持续消费消息总线中的消息"""
        self._running = True
        try:
            while self._running:
                msg = await self.bus.consume_inbound()
                asyncio.create_task(self._dispatch(msg))
        except asyncio.CancelledError:
            self._running = False
            raise

    async def _dispatch(self, msg: InboundMessage) -> None:
        """分发单条消息（带并发控制）"""
        async with self._concurrency_gate:
            session_key = msg.session_key
            async with self.get_session_lock(session_key):
                await self._process_message(msg)

    async def _process_message(self, msg: InboundMessage) -> None:
        """处理单条消息"""
        session = self.session_manager.get_session(msg.session_key)

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        history = truncate_history(history, max_messages=20)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        memory_ctx = self.memory.get_memory_context()
        if memory_ctx and messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n{memory_ctx}"

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        result = await self.runner.run(spec)

        session.add_message("user", msg.content)
        if result.final_content:
            session.add_message("assistant", result.final_content)

        self.memory.append_history({
            "session_key": msg.session_key,
            "user": msg.content[:200],
            "assistant": (result.final_content or "")[:200],
        })

        self.session_manager.save_session(session)

        response = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=result.final_content or "",
        )

        await self.bus.publish_outbound(response)

    async def chat(self, message: str, session_id: str = "default") -> str:
        """简单的单轮对话API（不经过消息总线）"""
        session = self.session_manager.get_session(session_id)

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        history = truncate_history(history, max_messages=20)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=message,
        )

        memory_ctx = self.memory.get_memory_context()
        if memory_ctx and messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n{memory_ctx}"

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        result = await self.runner.run(spec)

        session.add_message("user", message)
        if result.final_content:
            session.add_message("assistant", result.final_content)

        self.memory.append_history({
            "session_key": session_id,
            "user": message[:200],
            "assistant": (result.final_content or "")[:200],
        })

        self.session_manager.save_session(session)

        return result.final_content or ""

    def stop(self) -> None:
        """停止主循环"""
        self._running = False
