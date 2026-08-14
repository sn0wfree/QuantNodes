# coding=utf-8
"""VRP Harvesting: 波动率风险溢价收割.

当 VRP (IV - HV) 高时卖期权收 theta, 低时买期权.
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_put, select_call


@register
class VRPHarvesting(OptionStrategy):
    """VRP 收割: VRP 高→卖 put spread, VRP 低→买 straddle."""

    name = "vrp_harvesting"
    description = "VRP 高时卖 put spread 收 theta, VRP 低时买 straddle"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        sell_put_delta: float = 0.25,
        buy_put_delta: float = 0.10,
        buy_straddle_delta: float = 0.50,
        vrp_high_threshold: float = 0.6,
        vrp_low_threshold: float = 0.3,
    ):
        super().__init__(config)
        self._sell_put_delta = sell_put_delta
        self._buy_put_delta = buy_put_delta
        self._buy_straddle_delta = buy_straddle_delta
        self._vrp_high = vrp_high_threshold
        self._vrp_low = vrp_low_threshold

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")
        vrp_pct = kwargs.get("vrp_pct")  # VRP percentile (0~1)

        if existing_holding is not None:
            return {}

        if vrp_pct is None:
            return {}

        if vrp_pct >= self._vrp_high:
            # VRP 高: IV > HV, 期权贵 → 卖 put spread
            return self._sell_put_spread(product, date, underlying_price, loader, um, oe)
        elif vrp_pct <= self._vrp_low:
            # VRP 低: IV < HV, 期权便宜 → 买 straddle
            return self._buy_straddle(product, date, underlying_price, loader, um, oe)
        return {}

    def _sell_put_spread(self, product, date, spot, loader, um, oe):
        long_put = select_put(
            product, date, spot, target_delta=self._sell_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        short_put = select_put(
            product, date, spot, target_delta=self._buy_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if long_put is not None:
            targets[long_put.symbol] = 1.0
        if short_put is not None and short_put.strike < (long_put.strike if long_put else 0):
            targets[short_put.symbol] = -1.0
        return targets

    def _buy_straddle(self, product, date, spot, loader, um, oe):
        from ..selector import select_straddle
        s = select_straddle(
            product, date, spot,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        if s is not None:
            return {s.call.symbol: 1.0, s.put.symbol: 1.0}
        return {}
