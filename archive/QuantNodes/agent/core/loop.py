# coding=utf-8
"""
主循环 (AgentLoop)

消息总线驱动，处理并发会话
Phase A-F: Memory Persistence 集成
"""

import asyncio
import logging
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

logger = logging.getLogger(__name__)


class AgentLoop:
    """Agent主消息循环"""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path | str,
        model: str | None = None,
        max_iterations: int = 5,
        max_tokens: int = 102400,
        session_manager: SessionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        hook: AgentHook | None = None,
        mode_models: dict | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = Path(workspace)
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.hook = hook or CompositeHook()
        self.mode_models = mode_models or {}

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
        self.dream_store = DreamStore(self.workspace)
        self.dream_engine = DreamEngine(self.dream_store)

        # Phase F: 截断分析队列（per-session 隔离）
        self._pending_dream_analysis: Dict[str, List[Dict]] = {}
        self._compaction_counter: Dict[str, int] = {}

        self.runner = AgentRunner(provider, hook=self.hook)

        self._concurrency_gate = asyncio.Semaphore(1)
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._pending_queues: Dict[str, asyncio.Queue] = {}
        self._active_tasks: Dict[str, list[asyncio.Task]] = {}
        self._running = False

    def register_tool(self, tool: Any) -> None:
        """注册工具"""
        self.tool_registry.register(tool)

    def _estimate_tokens(self, messages: List[Dict[str, Any]], current_message: str = "") -> int:
        """估算消息列表的 token 数（粗略估算: 1 token ≈ 4 chars）"""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(str(part.get("text", "")))
        total_chars += len(current_message)
        return total_chars // 4

    def _get_context_limit(self, model: str | None) -> int:
        """获取模型的上下文窗口大小"""
        limits = {
            "minimax/minimax-m2.5:free": 1000000,
            "minimax/minimax-m2.5": 1000000,
            "minimax/minimax-m2.7": 1000000,
            "openai/gpt-4o": 128000,
            "openai/gpt-4o-mini": 128000,
            "anthropic/claude-3.5-sonnet": 200000,
        }
        return limits.get(model or "", 128000)

    def _auto_compact(self, messages: List[Dict[str, Any]], model: str | None) -> List[Dict[str, Any]]:
        """自动压缩：保留系统消息，将历史消息摘要化"""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        if not non_system_msgs:
            return messages

        # 保留最后 4 条消息，其余压缩为摘要
        keep_count = min(4, len(non_system_msgs))
        kept = non_system_msgs[-keep_count:]
        dropped = non_system_msgs[:-keep_count]

        if not dropped:
            return messages

        # 构建摘要文本（不调用 LLM，用简单截断）
        summary_parts = []
        for msg in dropped:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                summary_parts.append(f"[{role}]: {content[:200]}")

        summary = "Previous conversation summary:\n" + "\n".join(summary_parts)
        summary_msg = {"role": "system", "content": summary}

        return system_msgs + [summary_msg] + kept

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

        # Phase D: 注入 Dream 洞察
        dream_ctx = self.dream_store.get_injection_content(self.dream_engine.config)
        if dream_ctx:
            messages[0]["content"] += f"\n\n{dream_ctx}"

    async def _process_dream_analysis(
        self, session_key: str, user_content: str, final_content: str, result
    ) -> None:
        """Phase D + F: runner 完成后处理 Dream 分析"""
        session = self.session_manager.get_session(session_key)

        # Phase F: 处理待分析截断消息
        if session_key in self._pending_dream_analysis and self._pending_dream_analysis[session_key]:
            await self._process_compaction_dreams(session_key)

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

    async def _process_compaction_dreams(self, session_key: str) -> None:
        """分析被截断的消息，提取洞察生成 Dream"""
        pending = self._pending_dream_analysis.get(session_key, [])
        if not pending:
            return

        # 频率控制：每 N 次截断才分析一次
        counter = self._compaction_counter.get(session_key, 0) + 1
        self._compaction_counter[session_key] = counter
        if counter < self.dream_engine.config.compaction_dream_interval:
            self._pending_dream_analysis[session_key] = []
            return
        self._compaction_counter[session_key] = 0

        # 整体分析：将所有被丢弃的消息合并为一个摘要
        dropped_text = "\n".join([
            f"[{m.get('role', '?')}]: {m.get('content', '')[:200]}"
            for m in pending[-10:]
        ])

        # 快速检查是否包含可提取的洞察
        if not self.dream_engine.should_analyze_conversation(dropped_text, ""):
            self._pending_dream_analysis[session_key] = []
            return

        # 生成 Dream
        dream = await self.dream_engine.generate_dream(
            dream_type="compaction_insight",
            content=f"被截断的对话历史摘要 ({len(pending)} 条消息)",
            source="compaction",
            insights=[f"截断消息中检测到关键词，已提取摘要"],
            confidence=0.7,
            tags=["compaction", "auto"],
        )

        if dream and dream.confidence >= self.dream_engine.config.min_confidence:
            await self._update_dream_topic(dream)

        self._pending_dream_analysis[session_key] = []

    async def _process_message(self, msg: InboundMessage) -> None:
        """处理单条消息"""
        try:
            session = self.session_manager.get_session(msg.session_key)

            history = [
                m for m in session.messages
                if m.get("role") in ("user", "assistant", "tool")
            ]
            # Phase F: 截断前捕获被丢弃的消息
            history, dropped = truncate_history(history, max_messages=20)
            if dropped:
                if msg.session_key not in self._pending_dream_analysis:
                    self._pending_dream_analysis[msg.session_key] = []
                self._pending_dream_analysis[msg.session_key].extend(dropped)

            messages = self.context_builder.build_messages(
                history=history,
                current_message=msg.content,
                media=msg.media,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )

            # Phase C + B + D: 注入记忆上下文
            self._inject_memory_context(messages, msg.session_key)

            spec = AgentRunSpec(
                initial_messages=messages,
                tools=self.tool_registry,
                model=self.model,
                max_tokens=self.max_tokens,
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

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Error processing message from %s: %s", msg.session_key, e, exc_info=True,
            )
            error_response = OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"处理消息时发生错误: {str(e)}",
            )
            await self.bus.publish_outbound(error_response)

    async def chat(self, message: str, session_id: str = "default") -> str:
        """简单的单轮对话API（不经过消息总线）"""
        logger.info(f"[loop.chat] Called: session_id={session_id}, message_length={len(message)}")

        if self.provider is None:
            logger.warning("[loop.chat] Provider is None!")
            return "Error: LLM provider not configured."

        session = self.session_manager.get_session(session_id)
        logger.info(f"[loop.chat] Session obtained: {session_id}")

        history = [
            m for m in session.messages
            if m.get("role") in ("user", "assistant", "tool")
        ]
        logger.info(f"[loop.chat] History length: {len(history)}")

        # Phase F: 截断前捕获被丢弃的消息
        history, dropped = truncate_history(history, max_messages=20)
        if dropped:
            logger.info(f"[loop.chat] Dropped {len(dropped)} messages during truncation")

        messages = self.context_builder.build_messages(
            history=history,
            current_message=message,
        )
        logger.info(f"[loop.chat] Built messages count: {len(messages)}")

        # Phase C + B + D: 注入记忆上下文
        self._inject_memory_context(messages, session_id)

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=self.model,
            max_tokens=self.max_tokens,
            max_iterations=self.max_iterations,
        )
        logger.info(f"[loop.chat] Running agent spec...")

        result = await self.runner.run(spec)
        logger.info(f"[loop.chat] Runner completed: final_content_length={len(result.final_content) if result.final_content else 0}")

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
        logger.info(f"[loop.chat] Done: returning {len(result.final_content) if result.final_content else 0} chars")

        return result.final_content or ""

    async def chat_stream(self, message: str, session_id: str = "default", model: str | None = None, max_tokens: int | None = None):
        """流式单轮对话API（不经过消息总线）

        Args:
            message: 用户输入
            session_id: 会话ID
            model: 可选，覆盖本次对话使用的模型
            max_tokens: 可选，覆盖本次对话的最大token数

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
            if session_id not in self._pending_dream_analysis:
                self._pending_dream_analysis[session_id] = []
            self._pending_dream_analysis[session_id].extend(dropped)

        messages = self.context_builder.build_messages(
            history=history,
            current_message=message,
        )

        # Phase C + B + D: 注入记忆上下文
        self._inject_memory_context(messages, session_id)

        # Auto Compact: 检查是否需要压缩上下文
        estimated_tokens = self._estimate_tokens(messages)
        context_limit = self._get_context_limit(model or self.model)
        if estimated_tokens > context_limit * 0.9:
            yield {"type": "system", "content": f"Context approaching limit ({estimated_tokens}/{context_limit} tokens). Compacting..."}
            messages = self._auto_compact(messages, model or self.model)

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tool_registry,
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
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
