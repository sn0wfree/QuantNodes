# coding=utf-8
"""WorkflowRegistry — 注册表，管理所有可用 workflow。

Usage::

    from QuantNodes.agent.workflows.registry import REGISTRY, WorkflowSpec

    spec = WorkflowSpec(
        name="my-workflow",
        description="...",
        steps=[...],
        state_factory=lambda: MyState(),
        result_builder=lambda state, config: {...},
    )
    REGISTRY.register(spec)

    # 查找
    spec = REGISTRY.get("my-workflow")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .step_agent import StepAgentSpec

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSpec:
    """一个完整 workflow 的规格定义。"""

    name: str
    """workflow 名称，如 "alpha-gpt"。"""

    description: str
    """给 LLM 看的描述，包含配置参数说明。"""

    steps: list[StepAgentSpec]
    """每轮执行的步骤列表。"""

    state_factory: Callable[[], Any]
    """构造 state 对象的工厂函数。"""

    result_builder: Callable[[Any, dict], dict]
    """(state, config) -> result dict。构建最终结果。"""

    iterations: int = 1
    """轮数。1 = 线性 pipeline，>1 = 多轮迭代。"""

    final_steps: list[StepAgentSpec] = field(default_factory=list)
    """最终步骤列表（如 critic）。在所有轮次后执行。"""

    provider: Optional[str] = None
    """可选: 覆盖 LLM provider 名称。v2 扩展。"""

    model: Optional[str] = None
    """可选: 覆盖 model 名称。v2 扩展。"""


class WorkflowRegistry:
    """注册表，管理所有可用 workflow。"""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowSpec] = {}

    def register(self, spec: WorkflowSpec) -> None:
        """注册一个 workflow。"""
        if spec.name in self._workflows:
            logger.warning("WorkflowRegistry: overwriting existing workflow %r", spec.name)
        self._workflows[spec.name] = spec
        logger.info("WorkflowRegistry: registered workflow %r", spec.name)

    def get(self, name: str) -> Optional[WorkflowSpec]:
        """按名称查找 workflow。"""
        return self._workflows.get(name)

    def list_all(self) -> list[dict[str, str]]:
        """返回所有 workflow 的 name + description。"""
        return [
            {"name": spec.name, "description": spec.description}
            for spec in self._workflows.values()
        ]

    def build_llm_description(self) -> str:
        """拼接给 LLM 的 tool description。"""
        if not self._workflows:
            return "No workflows registered."

        lines = [
            "Execute a named programmatic workflow pipeline.",
            "",
            "Available workflows:",
        ]
        for spec in self._workflows.values():
            steps_info = []
            if spec.steps:
                steps_info.append(f"{len(spec.steps)} steps/round")
            if spec.iterations > 1:
                steps_info.append(f"{spec.iterations} iterations")
            if spec.final_steps:
                steps_info.append(f"{len(spec.final_steps)} final steps")
            meta = f" ({', '.join(steps_info)})" if steps_info else ""
            lines.append(f"- {spec.name}{meta}: {spec.description}")

        return "\n".join(lines)


# 模块级单例
REGISTRY = WorkflowRegistry()

__all__ = [
    "WorkflowSpec",
    "WorkflowRegistry",
    "REGISTRY",
]
