# coding=utf-8
"""
主循环 (AgentLoop)

消息总线驱动，处理并发会话
Phase A-F: Memory Persistence 集成
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List

from ..bus.events import InboundMessage, OutboundMessage
from ..bus.queue import MessageBus
from ..session.manager import SessionManager
from ..providers.base import LLMProvider
from ..tools.registry import ToolRegistry
from .context import ContextBuilder
from .runner import AgentRunner, AgentRunSpec
from .hook import AgentHook, CompositeHook
from .memory import MemoryStore, MemoryManager, DreamStore
from .dream import DreamEngine
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

        # Phase B: MemoryStore (enhanced append_history)
        self.memory = MemoryStore(self.workspace)

        # Phase C: MemoryManager (Claude Code style)
        self.memory_manager = MemoryManager(self.workspace)

        # Phase D: DreamEngine
        dream_store = DreamStore(self.workspace)
        self.dream_engine = DreamEngine(dream_store)

        # Phase F: 截断分析队列
        self._pending_dream_analysis: List[Dict] = []
        self._compaction_counter: int = 0

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

    def _inject_memory_context(self, messages: List[Dict], session_key: str = None) -> None:
        """注入记忆上下文到 system message（Phase C + Phase B）"""
        if not messages or messages[0].get("role") != "system":
            return

        # Phase C: 注入记忆索引
        memory_ctx = self.memory_manager.get_memory_context()
        if memory_ctx:
            messages[0]["content"] += f"\n\n{memory_ctx}"

        # Phase B: 注入最近对话摘要
        if session_key:
            recent = self.memory.get_recent_history(limit=5, session_key=session_key)
            if recent:
                summary_lines = [
                    f"- {r['user'][:80]} → {(r.get('assistant') or '')[:80]}"
                    for r in recent[-3:]
                ]
                summary = "\n".join(summary_lines)
                messages[0]["content"] += f"\n\n## 最近对话\n{summary}"

    async def _process_dream_analysis(
        self, session_key: str, user_content: str, final_content: str, result
    ) -> None:
        """Phase D + F: runner 完成后处理 Dream 分析"""
        session = self.session_manager.get_session(session_key)

        # Phase F: 处理待分析截断消息
        if self._pending_dream_analysis:
            await self._process_compaction_dreams()

        # Phase D: 对话洞察分析（仅在积累足够后触发）
        round_count = len([m for m in session.messages if m.get("role") == "user"])
        if round_count >= self.dream_engine.config.min_rounds_before_activate:
            if self.dream_engine.should_analyze_conversation(user_content, final_content or ""):
                dream = await self.dream_engine.analyze_conversation(
                    user_message=user_content,
                    assistant_response=final_content or "",
                    tools_used=result.tools_used if hasattr(result, 'tools_used') else [],
                )
                if dream and dream.confidence >= self.dream_engine.config.min_confidence:
                    await self._update_dream_topic(dream)

    async def _update_dream_topic(self, dream) -> None:
        """将 Dream 洞察写入 dream-insights.md 主题文件"""
        try:
            existing = self.memory_manager.read_topic("dream-insights")
            timestamp = dream.timestamp[:10]
            new_entry = f"\n### {timestamp} - {dream.type}\n"
            new_entry += f"- {dream.content}\n"
            for insight in dream.insights:
                new_entry += f"  - {insight}\n"

            updated = existing + new_entry
            self.memory_manager.write_topic("dream-insights", updated)

            # 更新 MEMORY.md 索引
            index = self.memory_manager.read_index()
            if "topic-dream-insights.md" not in index:
                index += "\n## Insights\n- Dream洞察记录 (topic-dream-insights.md)\n"
                self.memory_manager.write_index(index)
        except Exception:
            pass

    async def _process_compaction_dreams(self) -> None:
        """分析被截断的消息，提取洞察生成 Dream"""
        if not self._pending_dream_analysis:
            return

        # 频率控制：每 N 次截断才分析一次
        self._compaction_counter += 1
        if self._compaction_counter < self.dream_engine.config.compaction_dream_interval:
            self._pending_dream_analysis.clear()
            return
        self._compaction_counter = 0

        # 整体分析：将所有被丢弃的消息合并为一个摘要
        dropped_text = "\n".join([
            f"[{m.get('role', '?')}]: {m.get('content', '')[:200]}"
            for m in self._pending_dream_analysis[-10:]
        ])

        # 快速检查是否包含可提取的洞察
        if not self.dream_engine.should_analyze_conversation(dropped_text, ""):
            self._pending_dream_analysis.clear()
            return

        # 生成 Dream
        dream = await self.dream_engine.generate_dream(
            dream_type="compaction_insight",
            content=f"被截断的对话历史摘要 ({len(self._pending_dream_analysis)} 条消息)",
            source="compaction",
            insights=[f"截断消息中检测到关键词，已提取摘要"],
            confidence=0.7,
            tags=["compaction", "auto"],
        )

        if dream and dream.confidence >= self.dream_engine.config.min_confidence:
            await self._update_dream_topic(dream)

        self._pending_dream_analysis.clear()

    async def _process_message(self, msg: InboundMessage) -> None:
        """处理单条消息"""
        session = self.session_manager.get_session(msg.session_key)

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        # Phase F: 截断前捕获被丢弃的消息
        history, dropped = truncate_history(history, max_messages=20)
        if dropped:
            self._pending_dream_analysis.extend(dropped)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # Phase C + B: 注入记忆上下文
        self._inject_memory_context(messages, msg.session_key)

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        result = await self.runner.run(spec)

        # Phase D + F: Dream 分析
        await self._process_dream_analysis(
            msg.session_key, msg.content, result.final_content, result
        )

        session.add_message("user", msg.content)
        if result.final_content:
            session.add_message("assistant", result.final_content)

        self.memory.append_history(
            {
                "session_key": msg.session_key,
                "user": msg.content[:500],
                "assistant": (result.final_content or "")[:500],
            },
            tools_used=result.tools_used if hasattr(result, 'tools_used') else [],
        )

        self.session_manager.save_session(session)

        response = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=result.final_content or "",
        )

        await self.bus.publish_outbound(response)

    async def chat(self, message: str, session_id: str = "default") -> str:
        """简单的单轮对话API（不经过消息总线）"""
        if self.provider is None:
            return "Error: LLM provider not configured."

        session = self.session_manager.get_session(session_id)

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        # Phase F: 截断前捕获被丢弃的消息
        history, dropped = truncate_history(history, max_messages=20)
        if dropped:
            self._pending_dream_analysis.extend(dropped)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=message,
        )

        # Phase C + B: 注入记忆上下文
        self._inject_memory_context(messages, session_id)

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        result = await self.runner.run(spec)

        # Phase D + F: Dream 分析
        await self._process_dream_analysis(
            session_id, message, result.final_content, result
        )

        session.add_message("user", message)
        if result.final_content:
            session.add_message("assistant", result.final_content)

        self.memory.append_history(
            {
                "session_key": session_id,
                "user": message[:500],
                "assistant": (result.final_content or "")[:500],
            },
        )

        self.session_manager.save_session(session)

        return result.final_content or ""

    async def chat_stream(self, message: str, session_id: str = "default"):
        """流式单轮对话API（不经过消息总线）

        Yields:
            dict: 事件字典（token, tool_call, tool_result, done, error）
        """
        if self.provider is None:
            yield {"type": "error", "content": "LLM provider not configured."}
            return

        session = self.session_manager.get_session(session_id)

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        # Phase F: 截断前捕获被丢弃的消息
        history, dropped = truncate_history(history, max_messages=20)
        if dropped:
            self._pending_dream_analysis.extend(dropped)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=message,
        )

        # Phase C + B: 注入记忆上下文
        self._inject_memory_context(messages, session_id)

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        final_content = ""
        tools_used = []
        async for event in self.runner.run_stream(spec):
            if event["type"] == "done":
                final_content = event.get("content", "")
                tools_used = event.get("tools_used", [])
            yield event

        # Phase D + F: Dream 分析
        await self._process_dream_analysis(
            session_id, message, final_content,
            type('Result', (), {'final_content': final_content, 'tools_used': tools_used})()
        )

        session.add_message("user", message)
        if final_content:
            session.add_message("assistant", final_content)

        self.memory.append_history(
            {
                "session_key": session_id,
                "user": message[:500],
                "assistant": final_content[:500],
            },
        )

        self.session_manager.save_session(session)

    def stop(self) -> None:
        """停止主循环"""
        self._running = False
