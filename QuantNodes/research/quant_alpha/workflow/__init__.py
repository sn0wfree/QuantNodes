# coding=utf-8
"""
workflow - Alpha-GPT 5 智能体编排协调器（M5 PR）

详见 docs/quant_alpha/alpha_gpt_architecture.md
"""

from .state import (
    IdeaRecord,
    FormulaRecord,
    EvaluationRecord,
    ReflectionRecord,
    FinalFormulaRecord,
    AlphaGptState,
)
from .alpha_gpt import (
    AlphaGptConfig,
    AlphaGptResult,
    AlphaGptWorkflow,
)

__all__ = [
    "AlphaGptConfig",
    "AlphaGptResult",
    "AlphaGptWorkflow",
    "AlphaGptState",
    "IdeaRecord",
    "FormulaRecord",
    "EvaluationRecord",
    "ReflectionRecord",
    "FinalFormulaRecord",
]
