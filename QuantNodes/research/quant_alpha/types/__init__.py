# coding=utf-8
"""
types/ - 共享数据类型（叶子包，无外部依赖）

供 agent/workflows 和 research.quant_alpha 共同导入，
打破 agent ↔ research.quant_alpha 循环依赖。
"""

from QuantNodes.research.quant_alpha.types.state import (
    AlphaGptState,
    EvaluationRecord,
    FinalFormulaRecord,
    FormulaRecord,
    IdeaRecord,
    ReflectionRecord,
)
from QuantNodes.research.quant_alpha.types.constants import ALLOWED_OPERATORS

__all__ = [
    "AlphaGptState",
    "EvaluationRecord",
    "FinalFormulaRecord",
    "FormulaRecord",
    "IdeaRecord",
    "ReflectionRecord",
    "ALLOWED_OPERATORS",
]
