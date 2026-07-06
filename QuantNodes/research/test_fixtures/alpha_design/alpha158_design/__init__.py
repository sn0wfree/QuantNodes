# coding=utf-8
"""
alpha158_design - Alpha 158/360 设计哲学借鉴（M3 PR）

Alpha 158/360 (Yang et al. 2020, arXiv:2009.11189) 是 Qlib 平台的标准化
ML 特征集。本子包**借鉴其特征设计思想**而非直接移植公式集
（实际因子集由 ``QuantNodes.research.codegen`` 在他处生成）。

内容：
- DESIGN_PHILOSOPHY: Alpha 158/360 特征设计原则（4 类 × 7 条）
- CATEGORY_TEMPLATES: 4 类特征的设计模板（158 公式骨架 + 360 公式骨架）
- FEW_SHOT_EXAMPLES: 5-10 个示例特征（用于 Alpha-GPT 启动 prompt）

参考：
- Qlib: https://github.com/microsoft/qlib
- Alpha 158 = 9 KBAR + 20 Price + 5 Volume + 124 Rolling = 158
- Alpha 360 = 6 字段 × 60 lookback = 360

M3 范围：仅做借鉴（设计文档 + few-shot）。
完整特征集实现 → 路线 6 (Alpha-GPT) 阶段。
"""

from __future__ import annotations

from QuantNodes.research.test_fixtures.alpha_design.alpha158_design.philosophy import (
    FEATURE_CATEGORIES,
    CategoryTemplate,
    Alpha360Template,
    ALPHA360_TEMPLATE,
    DEFAULT_WINDOWS,
    ALPHA360_LOOKBACK_RANGE,
    get_template_by_category,
    get_template_by_name,
    list_categories,
    total_feature_count,
)
from QuantNodes.research.test_fixtures.alpha_design.alpha158_design.few_shot_examples import (
    ALPHA158_FEW_SHOT_EXAMPLES,
    list_examples,
    get_example,
    get_few_shot_prompt,
    get_categories as get_example_categories,
)

__all__ = [
    # 设计模板
    "FEATURE_CATEGORIES",
    "CategoryTemplate",
    "Alpha360Template",
    "ALPHA360_TEMPLATE",
    "DEFAULT_WINDOWS",
    "ALPHA360_LOOKBACK_RANGE",
    "get_template_by_category",
    "get_template_by_name",
    "list_categories",
    "total_feature_count",
    # few-shot 示例
    "ALPHA158_FEW_SHOT_EXAMPLES",
    "list_examples",
    "get_example",
    "get_few_shot_prompt",
    "get_example_categories",
]
