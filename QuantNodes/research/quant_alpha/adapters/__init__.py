# coding=utf-8
"""
adapters - AlphaGen 兼容适配器

让任何 RL 训练（AlphaGen / Alpha² / 自研）通过标准接口
使用 QuantNodes 算子 + factor_test 评估。

参考：AlphaGen (KDD 2023) 抽象接口 AlphaCalculator

M4 范围：
- Expression AST（极简子集，11 个算子）
- BaseAlphaCalculator ABC（7 个抽象方法）
- PolarsAlphaCalculator（参考实现，用 polars + OperatorVocab）
- expression_to_formula 转换器

后续路线（Phase 2）：
- 全栈 AlphaGen 复刻（40+ 人天）
- Alpha² 完整实现（45+ 人天）
- 自主 RL 训练（基于 PolarsAlphaCalculator）
"""

from __future__ import annotations

from QuantNodes.research.quant_alpha.adapters.calculator import (
    BaseAlphaCalculator,
    PolarsAlphaCalculator,
)
from QuantNodes.research.quant_alpha.adapters.expression import (
    # 基础
    Expression,
    Feature,
    Literal,
    Ref,
    # 二元
    BinaryOp,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    # 一元
    UnaryOp,
    Abs,
    Neg,
    Log,
    Sign,
    Sqrt,
    # 滚动
    RollingOp,
    Mean,
    Std,
    Sum,
    Max,
    Min,
    Delta,
    # 工具
    expression_to_formula,
    collect_feature_fields,
    collect_rolling_windows,
)

__all__ = [
    # ABC
    "BaseAlphaCalculator",
    "PolarsAlphaCalculator",
    # Expression 基类
    "Expression",
    "Feature",
    "Literal",
    "Ref",
    # 二元
    "BinaryOp",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Greater",
    "Less",
    # 一元
    "UnaryOp",
    "Abs",
    "Neg",
    "Log",
    "Sign",
    "Sqrt",
    # 滚动
    "RollingOp",
    "Mean",
    "Std",
    "Sum",
    "Max",
    "Min",
    "Delta",
    # 工具
    "expression_to_formula",
    "collect_feature_fields",
    "collect_rolling_windows",
]
