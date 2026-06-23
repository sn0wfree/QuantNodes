# coding=utf-8
"""
alpha101_design - Alpha 101 设计哲学借鉴（M3 PR）

Alpha 101 (Kakushadze 2015, arXiv:1601.00991) 提供了 101 个公式化 alpha
因子的设计范式。本子包**借鉴其设计哲学**而非直接移植公式集
（实际因子集由 llmwikify 在他处生成）。

内容：
- DESIGN_PHILOSOPHY: Alpha 101 核心设计原则（8 条）
- CORE_OPERATORS: 10-20 个核心算子子集（经济意义）
- FEW_SHOT_EXAMPLES: 5-10 个示例公式（用于 Alpha-GPT 启动 prompt）

M3 范围：仅做借鉴（设计文档 + few-shot）。
完整因子集实现 → 路线 6 (Alpha-GPT) 阶段。
"""

from __future__ import annotations

from QuantNodes.research.quant_alpha.alpha101_design.philosophy import (
    DESIGN_PHILOSOPHY,
    CORE_OPERATORS,
    A_SHARE_COMPATIBILITY,
    get_philosophy_by_id,
    get_operator_by_name,
    get_a_share_compatible_count,
)
from QuantNodes.research.quant_alpha.alpha101_design.few_shot_examples import (
    ALPHA101_FEW_SHOT_EXAMPLES,
    list_examples,
    get_example,
    get_few_shot_prompt,
    get_categories,
)

__all__ = [
    # 设计哲学
    "DESIGN_PHILOSOPHY",
    "CORE_OPERATORS",
    "A_SHARE_COMPATIBILITY",
    "get_philosophy_by_id",
    "get_operator_by_name",
    "get_a_share_compatible_count",
    # few-shot 示例
    "ALPHA101_FEW_SHOT_EXAMPLES",
    "list_examples",
    "get_example",
    "get_few_shot_prompt",
    "get_categories",
]
