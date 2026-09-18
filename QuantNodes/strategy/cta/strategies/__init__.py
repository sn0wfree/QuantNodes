# coding=utf-8
"""CTA 策略实现.

    - TrendFollowingStrategy: 多品种时序动量趋势跟踪 (策略 v1)
    - HuataiTimingCTAStrategy: 华泰时序CTA方法论落地 (策略 v6, IF/AU/T 期货实证)
    - run_ablations: 消融研究 (3 对照变体 + 参数预设批量跑批)
"""
from __future__ import annotations

from .trend_following import (
    DEFAULT_PRODUCTS,
    TrendFollowingConfig,
    TrendFollowingStrategy,
)
from .huatai_timing_cta import (
    HUATAI_PRODUCTS,
    HuataiTimingConfig,
    HuataiTimingCTAStrategy,
)
from .huatai_controls import run_ablations
from .portfolio_construction import (
    DEFAULT_DIVERSIFIED_POOL,
    LIQUIDITY_TOP_N,
    PERIOD_PRESETS,
    SECTOR_PRODUCTS,
    combine_period_navs,
    is_oos_validation_ratio,
    make_period_config,
    pool_sector_breakdown,
    sector_diversified_pool,
    split_is_oos,
)

__all__ = [
    "DEFAULT_DIVERSIFIED_POOL",
    "DEFAULT_PRODUCTS",
    "LIQUIDITY_TOP_N",
    "PERIOD_PRESETS",
    "SECTOR_PRODUCTS",
    "TrendFollowingConfig",
    "TrendFollowingStrategy",
    "HUATAI_PRODUCTS",
    "HuataiTimingConfig",
    "HuataiTimingCTAStrategy",
    "combine_period_navs",
    "is_oos_validation_ratio",
    "make_period_config",
    "pool_sector_breakdown",
    "run_ablations",
    "sector_diversified_pool",
    "split_is_oos",
]
