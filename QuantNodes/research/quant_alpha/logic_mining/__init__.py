# coding=utf-8
"""
logic_mining - 市场逻辑驱动的因子挖掘模块

基于 AlphaLogics 论文 (arXiv 2603.20247) 实现。

核心组件:
- WikiLogicStructured: 逻辑结构化表示
- CompiledConstraint (Γ): 编译后的可执行约束
- compile_to_constraint(): 逻辑 → Γ 编译器
"""

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.compiler import (
    CompiledConstraint,
    compile_to_constraint,
)

__all__ = [
    "LogicCondition",
    "LogicBehavior",
    "WikiLogicStructured",
    "CompiledConstraint",
    "compile_to_constraint",
]
