# coding=utf-8
"""v3 = Stage 16A 多策略组合 (新建于 2026-07-09).

特性:
  - SubStrategy 抽象基类 (sub_strategy_v3.py)
  - 多策略组合: 动量 + 均值反转 + 行业轮动
  - 子策略权重: 风险平价 / 等权
  - 预计 Calmar 1.70+, DD 进一步降低

v1 (原始CICC复现) 在 ../v1/ (Stage 8).
v2 (Stage 12A 增强版) 在 ../v2/ (已冻结).
v3 (Stage 16A 多策略) 在 ./ (新建).

参考: reports/momentum_etf_rotation/v2/stage16a_plan.md.
"""
from __future__ import annotations

from ..common.universe import (
    Category,
    DEFAULT_POOL,
    ETFCategorizer,
    ETFMeta,
    ETFPool,
)
from .sub_strategy_v3 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
    validate_sub_strategy_result,
)
from .reversion_v3 import (
    ReversionConfig,
    ReversionSubStrategy,
    reversion_score,
)
from .industry_rotation_v3 import (
    IndustryRotationConfig,
    IndustryRotationSubStrategy,
    industry_rotation_score,
    get_industry_codes,
    get_rebalance_dates,
)
from .sub_weighting_v3 import (
    combine_sub_results,
    equal_sub_weights,
    risk_parity_sub_weights,
    signal_weighted_sub_weights,
    sub_weights_from_results,
)


__all__ = [
    # Common re-exports
    "Category",
    "DEFAULT_POOL",
    "ETFCategorizer",
    "ETFMeta",
    "ETFPool",
    # SubStrategy (v3 specific)
    "SubStrategy",
    "SubStrategyConfig",
    "SubStrategyResult",
    "validate_sub_strategy_result",
    # Reversion (v3 specific)
    "ReversionConfig",
    "ReversionSubStrategy",
    "reversion_score",
    # Industry Rotation (v3 specific)
    "IndustryRotationConfig",
    "IndustryRotationSubStrategy",
    "industry_rotation_score",
    "get_industry_codes",
    "get_rebalance_dates",
    # Sub-Strategy Weighting (v3 specific)
    "equal_sub_weights",
    "sub_weights_from_results",
    "risk_parity_sub_weights",
    "signal_weighted_sub_weights",
    "combine_sub_results",
]
