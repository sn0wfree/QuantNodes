# coding=utf-8
"""Short Put Spread / Short Call Spread: 有限风险的卖方策略.

Short Put Spread: 卖高 delta put + 买低 delta put (看不跌)
Short Call Spread: 卖低 delta call + 买高 delta call (看不涨)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_put, select_call


@register
class ShortPutSpread(OptionStrategy):
    """卖 Put Spread: 卖高 delta put + 买低 delta put."""

    name = "short_put_spread"
    description = "卖 put spread, 收取权利金, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        sell_delta: float = 0.25,
        buy_delta: float = 0.10,
    ):
        super().__init__(config)
        self._sell_delta = sell_delta
        self._buy_delta = buy_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        sell_put = select_put(
            product, date, underlying_price, target_delta=self._sell_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        buy_put = select_put(
            product, date, underlying_price, target_delta=self._buy_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if sell_put is not None:
            targets[sell_put.symbol] = -1.0  # 卖高 delta put
        if buy_put is not None and buy_put.strike < (sell_put.strike if sell_put else 0):
            targets[buy_put.symbol] = 1.0    # 买低 delta put
        return targets


@register
class ShortCallSpread(OptionStrategy):
    """卖 Call Spread: 卖低 delta call + 买高 delta call."""

    name = "short_call_spread"
    description = "卖 call spread, 收取权利金, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        sell_delta: float = 0.25,
        buy_delta: float = 0.10,
    ):
        super().__init__(config)
        self._sell_delta = sell_delta
        self._buy_delta = buy_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        sell_call = select_call(
            product, date, underlying_price, target_delta=self._sell_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        buy_call = select_call(
            product, date, underlying_price, target_delta=self._buy_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if sell_call is not None:
            targets[sell_call.symbol] = -1.0  # 卖低 delta call
        if buy_call is not None and buy_call.strike > (sell_call.strike if sell_call else 0):
            targets[buy_call.symbol] = 1.0    # 买高 delta call
        return targets
