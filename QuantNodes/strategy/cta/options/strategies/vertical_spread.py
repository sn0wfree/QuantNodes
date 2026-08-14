# coding=utf-8
"""Vertical Spread: 买高 delta + 卖低 delta (同方向, 同到期).

牛市看涨: 买 call 低 strike + 卖 call 高 strike
熊市看跌: 买 put 高 strike + 卖 put 低 strike
风险: 封顶 (最大亏损 = 净权利金)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class BullCallSpread(OptionStrategy):
    """牛市看涨价差: 买低 strike call + 卖高 strike call."""

    name = "bull_call_spread"
    description = "买 ITM call + 卖 OTM call, 看涨, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        buy_delta: float = 0.50,
        sell_delta: float = 0.25,
    ):
        super().__init__(config)
        self._buy_delta = buy_delta
        self._sell_delta = sell_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        buy_call = select_call(
            product, date, underlying_price,
            target_delta=self._buy_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        sell_call = select_call(
            product, date, underlying_price,
            target_delta=self._sell_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if buy_call is not None:
            targets[buy_call.symbol] = 1.0
        if sell_call is not None and sell_call.strike > (buy_call.strike if buy_call else 0):
            targets[sell_call.symbol] = -1.0
        return targets


@register
class BearPutSpread(OptionStrategy):
    """熊市看跌价差: 买高 strike put + 卖低 strike put."""

    name = "bear_put_spread"
    description = "买 ITM put + 卖 OTM put, 看跌, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        buy_delta: float = 0.50,
        sell_delta: float = 0.25,
    ):
        super().__init__(config)
        self._buy_delta = buy_delta
        self._sell_delta = sell_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        buy_put = select_put(
            product, date, underlying_price,
            target_delta=self._buy_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        sell_put = select_put(
            product, date, underlying_price,
            target_delta=self._sell_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if buy_put is not None:
            targets[buy_put.symbol] = 1.0
        if sell_put is not None and sell_put.strike < (buy_put.strike if buy_put else float("inf")):
            targets[sell_put.symbol] = -1.0
        return targets
