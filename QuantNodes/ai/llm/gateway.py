# coding=utf-8
"""LLMGateway — 统一 LLM 调用入口。

所有模块通过 LLMGateway 调用 LLM，内部委托 nanobot Agent。

支持 4 种调用接口:
    - chat(messages) → ChatCompletion          # LLMClientBase 兼容
    - complete(agent_id, prompt) → str          # alpha_gpt 兼容
    - __call__(prompt) → str                    # callable 兼容
    - await run(prompt, session_id) → str       # nanobot 原生

支持工具调用:
    - tools: 工具名列表 (None=全部, []=无工具)
    - tool_choice: "auto"/"none"/"required"
    - with_tool_events: 异步接口返回 ToolCallResponse

支持重试机制和超时控制:
    - max_retries: 最大重试次数 (默认 3)
    - retry_delay: 重试间隔秒数 (默认 1.0)
    - timeout: 超时秒数 (默认 120)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from QuantNodes.ai.llm.base import (
    ChatCompletion,
    LLMClientBase,
    Message,
    MessageRole,
)

logger = logging.getLogger("llm.gateway")


@dataclass
class LLMConfig:
    """LLM 配置（重试机制和超时控制）"""
    max_retries: int = 3           # 最大重试次数
    retry_delay: float = 1.0       # 重试间隔秒数
    timeout: float = 300.0         # 超时秒数（MiniMax M3 复杂 JSON 需要较长时间）


@dataclass
class ToolCallResponse:
    """工具调用响应 (with_tool_events=True 时返回)。"""
    content: str
    tools_used: List[str] = field(default_factory=list)
    stop_reason: str = "stop"
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tools_used) > 0


class LLMGateway(LLMClientBase):
    """统一 LLM 调用门面 — 所有 LLM 调用的唯一入口。

    实现 LLMClientBase 接口 (chat / chat_stream)，同时提供:
    - complete(): alpha_gpt 兼容
    - __call__(): callable injection 兼容
    - run(): nanobot 原生 async

    支持工具调用:
    - tools: 工具名列表 (None=全部, []=无工具)
    - tool_choice: "auto"/"none"/"required"
    - with_tool_events: 异步接口返回 ToolCallResponse

    支持重试机制和超时控制:
    - max_retries: 最大重试次数 (默认 3)
    - retry_delay: 重试间隔秒数 (默认 1.0)
    - timeout: 超时秒数 (默认 120)

    Examples:
        >>> gateway = LLMGateway()
        >>> response = gateway.chat([Message(role="user", content="Hello")])
        >>> print(response.content)
        >>> response = gateway.chat(messages, tools=["backtest", "factor"])
        >>> print(response.tool_calls)
    """

    def __init__(
        self,
        agent: Any = None,
        workspace: str = ".agent",
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """初始化 LLMGateway。

        Args:
            agent: nanobot Agent 实例 (可选, 不传则自动创建)
            workspace: nanobot workspace 路径
            llm_config: LLM 配置（重试机制和超时控制）
        """
        super().__init__(**kwargs)
        self._agent = agent
        self._workspace = workspace
        self._agent_resolved = False
        self._llm_config = llm_config or LLMConfig()

    def _ensure_agent(self) -> Any:
        """懒加载 nanobot Agent。"""
        if self._agent is not None:
            return self._agent

        if not self._agent_resolved:
            self._agent_resolved = True
            try:
                from QuantNodes.agent.nanobot_bridge import Agent
                self._agent = Agent(workspace=self._workspace)
                # 覆盖 provider.generation.max_tokens（nanobot 默认 8192 太小）
                from dataclasses import replace
                loop = self._agent._loop
                gen = getattr(loop.provider, "generation", None)
                if gen is not None and gen.max_tokens < 16384:
                    loop.provider.generation = replace(gen, max_tokens=16384)
                    logger.info("LLMGateway: provider.generation.max_tokens → 16384")
                logger.info("LLMGateway: nanobot Agent 已初始化")
            except Exception as e:
                logger.warning("LLMGateway: nanobot 不可用, 降级到 NullLLMClient: %s", e)
                from QuantNodes.ai.llm.null import NullLLMClient
                self._agent = None
                self._fallback = NullLLMClient()

        return self._agent

    # ─── 接口 A: LLMClientBase.chat() 兼容 ───

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> ChatCompletion:
        """LLMClientBase 抽象方法实现。

        Args:
            messages: 对话消息列表
            model: 模型名称 (未使用, 由 nanobot 配置决定)
            tools: 工具名列表 (None=全部, []=无工具)
            tool_choice: "auto"/"none"/"required"

        Returns:
            ChatCompletion, 含:
            - content: 最终文本
            - tool_calls: [{id, name, arguments}, ...] (若 LLM 调了工具)
            - finish_reason: "stop" / "tool_calls"
        """
        prompt = self._messages_to_prompt(messages)
        agent = self._ensure_agent()

        if agent is None:
            return self._fallback._call_api(messages, model, **kwargs)

        result = self._run_sync(
            self._async_chat_collect(
                agent, prompt,
                tools=tools, tool_choice=tool_choice,
            )
        )
        return ChatCompletion(
            content=result["content"],
            role=MessageRole.ASSISTANT,
            finish_reason=result["stop_reason"],
            tool_calls=result["tool_calls"],
        )

    # ─── 接口 B: complete 兼容 ───

    def complete(
        self,
        agent_id: str = "default",
        prompt: str = "",
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """同步调用 LLM, 返回字符串结果。

        兼容 alpha_gpt.py 的 llm_client.complete(agent_id, prompt) 调用。

        Note: 同步接口返回纯文本 (工具调用已被 LLM 整合到 content 中)。
        如需工具调用详情, 请使用 chat() 或 run(with_tool_events=True)。

        支持重试机制和超时控制。
        """
        agent = self._ensure_agent()

        if agent is None:
            return self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            ).content

        # 重试机制
        last_error = None
        for attempt in range(self._llm_config.max_retries + 1):
            try:
                start_time = time.time()

                result = self._run_sync(
                    self._async_chat_collect(
                        agent, prompt,
                        session_id=agent_id,
                        tools=tools, tool_choice=tool_choice,
                        temperature=temperature,
                    )
                )

                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > self._llm_config.timeout:
                    raise TimeoutError(f"LLM call timed out after {elapsed:.1f}s")

                return result["content"] or ""

            except (TimeoutError, Exception) as e:
                last_error = e
                if attempt < self._llm_config.max_retries:
                    delay = self._llm_config.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._llm_config.max_retries, delay, str(e)[:100]
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        self._llm_config.max_retries + 1, str(e)[:200]
                    )

        # 所有重试都失败
        raise last_error

    # ─── 接口 C: callable 兼容 ───

    def __call__(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
    ) -> str:
        """callable 兼容: llm_judge / lineage_compress / operators 使用。"""
        agent = self._ensure_agent()

        if agent is None:
            return self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            ).content

        result = self._run_sync(
            self._async_chat_collect(
                agent, prompt,
                tools=tools, tool_choice=tool_choice,
            )
        )
        return result["content"] or ""

    # ─── 接口 D: nanobot 原生 async ───

    async def run(
        self,
        prompt: str,
        session_id: str = "default",
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
        with_tool_events: bool = False,
    ) -> Union[str, ToolCallResponse]:
        """异步调用 nanobot Agent。

        Args:
            prompt: 用户消息
            session_id: 会话 ID
            tools: 工具名列表 (None=全部, []=无工具)
            tool_choice: "auto"/"none"/"required"
            with_tool_events: True → 返回 ToolCallResponse (含工具调用详情)

        Returns:
            str: 默认行为, 仅返回最终文本
            ToolCallResponse: with_tool_events=True, 包含 tools_used/stop_reason/events
        """
        agent = self._ensure_agent()

        if agent is None:
            result = self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            )
            if with_tool_events:
                return ToolCallResponse(
                    content=result.content,
                    stop_reason="stop",
                )
            return result.content

        if with_tool_events:
            return await self._async_chat_collect(
                agent, prompt,
                session_id=session_id,
                tools=tools, tool_choice=tool_choice,
                collect_events=True,
            )

        result = await self._async_chat_collect(
            agent, prompt,
            session_id=session_id,
            tools=tools, tool_choice=tool_choice,
        )
        return result["content"] or ""

    # ─── 内部工具 ───

    async def _async_chat_collect(
        self,
        agent,
        prompt: str,
        session_id: str = "default",
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
        collect_events: bool = False,
        temperature: Optional[float] = None,
    ) -> Union[Dict[str, Any], ToolCallResponse]:
        """异步消费 agent.chat() 流, 收集最终结果 + 工具调用。"""
        final_content = ""
        tool_calls = []
        stop_reason = "stop"
        events = []

        async for event in agent.chat(
            prompt,
            session_id=session_id,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        ):
            etype = event.get("type")

            if collect_events:
                events.append(event)

            if etype == "tool_call":
                tool_calls.append({
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "arguments": event.get("arguments", {}),
                })
            elif etype == "done":
                final_content = event.get("content", final_content)
                stop_reason = event.get("stop_reason", "stop")
                if tool_calls:
                    stop_reason = "tool_calls"
            elif etype == "error":
                final_content = event.get("content", final_content)
                stop_reason = "error"

        if collect_events:
            return ToolCallResponse(
                content=final_content,
                tools_used=[tc["name"] for tc in tool_calls],
                stop_reason=stop_reason,
                events=events,
            )

        return {
            "content": final_content,
            "tool_calls": tool_calls if tool_calls else None,
            "stop_reason": stop_reason,
        }

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """将 Message 列表转为单一 prompt 字符串。"""
        parts = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                parts.append(f"[System] {msg.content}")
            elif msg.role == MessageRole.USER:
                parts.append(msg.content)
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"[Assistant] {msg.content}")
            else:
                parts.append(msg.content)
        return "\n\n".join(parts)

    def _run_sync(self, coro) -> Any:
        """在同步上下文中运行 async 协程。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)


# ─── 全局单例 ───

_global_gateway: Optional[LLMGateway] = None


def get_llm_gateway(workspace: str = ".agent") -> LLMGateway:
    """获取全局 LLMGateway 单例。"""
    global _global_gateway
    if _global_gateway is None:
        _global_gateway = LLMGateway(workspace=workspace)
    return _global_gateway


def create_llm_gateway(
    agent: Any = None, workspace: str = ".agent"
) -> LLMGateway:
    """创建新的 LLMGateway 实例（非单例）。"""
    return LLMGateway(agent=agent, workspace=workspace)


def reset_llm_gateway() -> None:
    """重置全局 LLMGateway 单例（测试用）。"""
    global _global_gateway
    _global_gateway = None
