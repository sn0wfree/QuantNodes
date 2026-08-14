# coding=utf-8
"""IV Timing: 低 IV 时买期权, 高 IV 时卖期权.

核心逻辑: IV 有均值回复特性, 低 IV 时 gamma 便宜, 高 IV 时 theta 丰厚.
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_straddle, select_put, select_call


@register
class IVTiming(OptionStrategy):
    """IV 择时: IV 低→买 straddle, IV 高→卖 strangle."""

    name = "iv_timing"
    description = "IV 低时买 straddle (便宜 gamma), IV 高时卖 strangle (贵 theta)"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        iv_low_pct: float = 0.3,
        iv_high_pct: float = 0.7,
        strangle_call_delta: float = 0.25,
        strangle_put_delta: float = 0.25,
    ):
        super().__init__(config)
        self._iv_low = iv_low_pct
        self._iv_high = iv_high_pct
        self._call_delta = strangle_call_delta
        self._put_delta = strangle_put_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")
        iv_pct = kwargs.get("iv_pct")  # IV percentile (0~1)

        if existing_holding is not None:
            return {}

        if iv_pct is None:
            return {}

        if iv_pct <= self._iv_low:
            # IV 低: 买 straddle (gamma 便宜)
            s = select_straddle(
                product, date, underlying_price,
                min_days=self._cfg.min_days, max_days=self._cfg.max_days,
                loader=loader, um=um, oe=oe,
            )
            if s is not None:
                return {s.call.symbol: 1.0, s.put.symbol: 1.0}
        elif iv_pct >= self._iv_high:
            # IV 高: 卖 strangle (theta 丰厚)
            call = select_call(
                product, date, underlying_price, target_delta=self._call_delta,
                min_days=self._cfg.min_days, max_days=self._cfg.max_days,
                loader=loader, um=um, oe=oe,
            )
            put = select_put(
                product, date, underlying_price, target_delta=self._put_delta,
                min_days=self._cfg.min_days, max_days=self._cfg.max_days,
                loader=loader, um=um, oe=oe,
            )
            targets = {}
            if call is not None:
                targets[call.symbol] = -1.0
            if put is not None:
                targets[put.symbol] = -1.0
            return targets
        return {}
