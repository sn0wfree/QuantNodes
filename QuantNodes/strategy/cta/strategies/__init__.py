# coding=utf-8
"""CTA 策略实现.

    - TrendFollowingStrategy: 多品种时序动量趋势跟踪 (策略 v1)
"""
from __future__ import annotations

from .trend_following import (
    DEFAULT_PRODUCTS,
    TrendFollowingConfig,
    TrendFollowingStrategy,
)

__all__ = [
    "DEFAULT_PRODUCTS",
    "TrendFollowingConfig",
    "TrendFollowingStrategy",
]
