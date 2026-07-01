# coding=utf-8
"""
state.py - Alpha-GPT workflow 状态管理

5 类记录的 dataclass + AlphaGptState 容器。

实际定义已迁移到 types/state.py（叶子包，打破循环依赖）。
本文件保持向后兼容 re-export。
"""

from QuantNodes.research.quant_alpha.types.state import (
    AlphaGptState,
    EvaluationRecord,
    FinalFormulaRecord,
    FormulaRecord,
    IdeaRecord,
    ReflectionRecord,
)

__all__ = [
    "IdeaRecord",
    "FormulaRecord",
    "EvaluationRecord",
    "ReflectionRecord",
    "FinalFormulaRecord",
    "AlphaGptState",
]
