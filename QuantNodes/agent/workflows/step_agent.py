# coding=utf-8
"""StepAgent — 轻量级单次 LLM 步骤。

无工具循环、无会话历史、无 Hook/Streaming。
用于 WorkflowTool 内部的 pipeline 步骤执行。

Usage::

    from QuantNodes.agent.workflows.step_agent import StepAgent, StepAgentSpec

    spec = StepAgentSpec(
        agent_id="my-step",
        prompt_builder=lambda **ctx: "do something",
        output_parser=lambda raw: ParseResult(ok=True, data=json.loads(raw)),
        output_key="results",
        record_factory=lambda d: d,
    )
    agent = StepAgent(spec, llm_client=my_provider)
    records = agent.run(state=state, round_idx=1, **config)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from QuantNodes.research.quant_alpha.llm.parser import ParseResult

logger = logging.getLogger(__name__)


@dataclass
class StepAgentSpec:
    """单个 pipeline 步骤的规格定义。"""

    agent_id: str
    """步骤标识，如 "alpha-gpt-idea-generator"。"""

    prompt_builder: Optional[Callable[..., str]] = None
    """(**ctx) -> prompt str。evaluator 步骤为 None。"""

    output_parser: Optional[Callable[[str], ParseResult]] = None
    """(raw) -> ParseResult。evaluator 步骤为 None。"""

    output_key: str = ""
    """JSON 输出的 key，如 "ideas", "formulas"。"""

    record_factory: Optional[Callable[[dict], Any]] = None
    """(dict) -> Record。evaluator 步骤为 None（tool_executor 直接返回 records）。"""

    state_output: Optional[str] = None
    """写入 state 的字段名，如 "all_ideas"。None = 不更新 state。"""

    state_input: Optional[str] = None
    """从 state 读取的字段名，注入到 prompt_builder 的 context。"""

    tool_executor: Optional[Callable[..., list[Any]]] = None
    """直接调用的工具函数（跳过 LLM）。evaluator 步骤用。"""

    max_retries: int = 2
    """解析失败重试次数。总共尝试 max_retries + 1 次。"""

    skip_on_last: bool = False
    """最后一轮是否跳过。reflector 步骤用。"""


class StepAgent:
    """轻量级单次 LLM 步骤。无工具循环、无会话历史。

    由 WorkflowTool 的 execute() 循环调用。
    """

    def __init__(self, spec: StepAgentSpec, llm_client: Any = None) -> None:
        self.spec = spec
        self.llm_client = llm_client

    def run(
        self,
        state: Any = None,
        round_idx: Optional[int] = None,
        prev_output: Optional[list[Any]] = None,
        **context: Any,
    ) -> list[Any]:
        """执行: prompt → LLM → parse(带重试+修复) → records。

        Args:
            state: workflow 状态对象。prompt_builder 可从中读取累积数据。
            round_idx: 当前轮次 (多轮 workflow 用)。
            prev_output: 上一步的输出 (轮内链式传递)。
            **context: 其他参数 (config 等)。

        Returns:
            解析后的 records 列表。失败时返回空列表。
        """
        # 从 state 读取 state_input
        if self.spec.state_input and state is not None:
            context[self.spec.state_input] = getattr(state, self.spec.state_input)

        # 轮内上一步输出
        if prev_output is not None:
            context["prev_output"] = prev_output

        context["state"] = state
        context["round_idx"] = round_idx

        # tool_executor 路径 (evaluator)
        if self.spec.tool_executor is not None:
            return self._run_tool(**context)

        # LLM 路径 (带重试)
        if self.spec.prompt_builder is None or self.spec.output_parser is None:
            logger.warning("StepAgent %s: no prompt_builder or output_parser", self.spec.agent_id)
            return []

        result = self._run_with_retry(**context)
        if not result.ok:
            logger.warning(
                "StepAgent %s: all %d attempts failed. Last error: %s",
                self.spec.agent_id,
                self.spec.max_retries + 1,
                result.error,
            )
            return []

        items = (result.data or {}).get(self.spec.output_key, [])
        if self.spec.record_factory is not None:
            return [self.spec.record_factory(item) for item in items]
        return items

    def _run_tool(self, **context: Any) -> list[Any]:
        """调用 tool_executor（同步或异步）。"""
        result = self.spec.tool_executor(**context)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return _run_async(result)
        return result

    def _run_with_retry(self, **context: Any) -> ParseResult:
        """重试逻辑: 解析失败时注入完整 raw + error 到 prompt。"""
        prompt = self.spec.prompt_builder(**context)
        result: Optional[ParseResult] = None

        for attempt in range(self.spec.max_retries + 1):
            raw = self._call_llm(prompt)
            result = self.spec.output_parser(raw)
            if result.ok:
                return result

            if attempt < self.spec.max_retries:
                logger.info(
                    "StepAgent %s: attempt %d/%d parse failed, retrying with error context",
                    self.spec.agent_id,
                    attempt + 1,
                    self.spec.max_retries + 1,
                )
                prompt = self.spec.prompt_builder(
                    **context,
                    _prev_error=result.error,
                    _prev_raw=result.raw,
                )

        return result  # type: ignore[return-value]

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM。

        优先级: client.complete(agent_id, prompt) → client(prompt) → mock。
        """
        client = self.llm_client
        if client is None:
            return self._mock_response(prompt)

        try:
            if hasattr(client, "complete"):
                return client.complete(agent_id=self.spec.agent_id, prompt=prompt)
            if callable(client):
                return client(prompt)
        except Exception as exc:
            logger.error("StepAgent %s: LLM call failed: %s", self.spec.agent_id, exc)
            return ""

        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """无 LLM 时返回空 JSON。测试用。"""
        return "{}"


def _run_async(coro: Any) -> Any:
    """在同步上下文中运行异步协程。

    处理三种情况:
    1. 没有 event loop → asyncio.run()
    2. 有 event loop 但不在其中 → loop.run_until_complete()
    3. 在 event loop 中（如 Jupyter）→ ThreadPoolExecutor
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or loop.is_closed():
        return asyncio.run(coro)

    # 在 event loop 中，用线程池绕过
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


__all__ = [
    "StepAgent",
    "StepAgentSpec",
    "ParseResult",
]
