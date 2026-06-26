# coding=utf-8
"""QuantNodes Workflows — 多智能体 WorkflowTool 框架。

声明式定义 pipeline 步骤 (StepAgentSpec)，
框架驱动多轮迭代执行 (WorkflowTool)。
"""

from .step_agent import ParseResult, StepAgent, StepAgentSpec
from .registry import REGISTRY, WorkflowRegistry, WorkflowSpec
from .tool import WorkflowTool

__all__ = [
    "StepAgent",
    "StepAgentSpec",
    "ParseResult",
    "WorkflowRegistry",
    "WorkflowSpec",
    "REGISTRY",
    "WorkflowTool",
]
