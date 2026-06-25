# coding=utf-8
"""LLMGateway — 统一 LLM 调用入口。

所有模块通过 LLMGateway 调用 LLM，内部委托 nanobot Agent。

支持 4 种调用接口:
    - chat(messages) → ChatCompletion          # LLMClientBase 兼容
    - complete(agent_id, prompt) → str          # alpha_gpt 兼容
    - __call__(prompt) → str                    # callable 兼容
    - await run(prompt, session_id) → str       # nanobot 原生
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from QuantNodes.ai.llm.base import (
    ChatCompletion,
    LLMClientBase,
    Message,
    MessageRole,
)

logger = logging.getLogger("llm.gateway")


class LLMGateway(LLMClientBase):
    """统一 LLM 调用门面 — 所有 LLM 调用的唯一入口。

    实现 LLMClientBase 接口 (chat / chat_stream)，同时提供:
    - complete(): alpha_gpt 兼容
    - __call__(): callable injection 兼容
    - run(): nanobot 原生 async

    Examples:
        >>> gateway = LLMGateway()
        >>> response = gateway.chat([Message(role="user", content="Hello")])
        >>> print(response.content)
    """

    def __init__(
        self,
        agent: Any = None,
        workspace: str = ".agent",
        **kwargs,
    ):
        """初始化 LLMGateway。

        Args:
            agent: nanobot Agent 实例 (可选, 不传则自动创建)
            workspace: nanobot workspace 路径
        """
        super().__init__(**kwargs)
        self._agent = agent
        self._workspace = workspace
        self._agent_resolved = False

    def _ensure_agent(self) -> Any:
        """懒加载 nanobot Agent。"""
        if self._agent is not None:
            return self._agent

        if not self._agent_resolved:
            self._agent_resolved = True
            try:
                from QuantNodes.agent.nanobot_bridge import Agent
                self._agent = Agent(workspace=self._workspace)
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
        **kwargs,
    ) -> ChatCompletion:
        """LLMClientBase 抽象方法实现。"""
        prompt = self._messages_to_prompt(messages)
        agent = self._ensure_agent()

        if agent is None:
            # 降级: 使用 NullLLMClient
            return self._fallback._call_api(messages, model, **kwargs)

        result = self._run_sync(
            agent.run(prompt, session_id="default")
        )
        return ChatCompletion(
            content=result or "",
            role=MessageRole.ASSISTANT,
        )

    # ─── 接口 B: complete 兼容 ───

    def complete(self, agent_id: str = "default", prompt: str = "") -> str:
        """调用 LLM，返回字符串结果。

        兼容 alpha_gpt.py 的 llm_client.complete(agent_id, prompt) 调用。
        """
        agent = self._ensure_agent()

        if agent is None:
            return self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            ).content

        return self._run_sync(
            agent.run(prompt, session_id=agent_id)
        ) or ""

    # ─── 接口 C: callable 兼容 ───

    def __call__(self, prompt: str) -> str:
        """ callable 兼容: llm_judge / lineage_compress / operators 使用。"""
        agent = self._ensure_agent()

        if agent is None:
            return self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            ).content

        return self._run_sync(
            agent.run(prompt, session_id="default")
        ) or ""

    # ─── 接口 D: nanobot 原生 async ───

    async def run(self, prompt: str, session_id: str = "default") -> str:
        """异步调用 nanobot Agent.run()。"""
        agent = self._ensure_agent()

        if agent is None:
            result = self._fallback._call_api(
                [Message(role=MessageRole.USER, content=prompt)]
            )
            return result.content

        return await agent.run(prompt, session_id=session_id) or ""

    # ─── 内部工具 ───

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
