# coding=utf-8
"""Diagonal Spread: 不同 strike + 不同到期.

牛市看涨 Diagonal: 买近月低 strike call + 卖远月高 strike call
熊市看跌 Diagonal: 买近月高 strike put + 卖远月低 strike put
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class BullCallDiagonal(OptionStrategy):
    """牛市看涨 Diagonal: 买近月低 strike call + 卖远月高 strike call."""

    name = "bull_call_diagonal"
    description = "买近月ATM call + 卖远月OTM call, 看涨 + theta"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        buy_delta: float = 0.50,
        sell_delta: float = 0.25,
        near_min_days: int = 15,
        near_max_days: int = 35,
        far_min_days: int = 45,
        far_max_days: int = 75,
    ):
        super().__init__(config)
        self._buy_delta = buy_delta
        self._sell_delta = sell_delta
        self._near_min = near_min_days
        self._near_max = near_max_days
        self._far_min = far_min_days
        self._far_max = far_max_days

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        buy_call = select_call(
            product, date, underlying_price, target_delta=self._buy_delta,
            min_days=self._near_min, max_days=self._near_max,
            loader=loader, um=um, oe=oe,
        )
        sell_call = select_call(
            product, date, underlying_price, target_delta=self._sell_delta,
            min_days=self._far_min, max_days=self._far_max,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if buy_call is not None:
            targets[buy_call.symbol] = 1.0   # 买近月
        if sell_call is not None:
            targets[sell_call.symbol] = -1.0  # 卖远月
        return targets


@register
class BearPutDiagonal(OptionStrategy):
    """熊市看跌 Diagonal: 买近月高 strike put + 卖远月低 strike put."""

    name = "bear_put_diagonal"
    description = "买近月ATM put + 卖远月OTM put, 看跌 + theta"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        buy_delta: float = 0.50,
        sell_delta: float = 0.25,
        near_min_days: int = 15,
        near_max_days: int = 35,
        far_min_days: int = 45,
        far_max_days: int = 75,
    ):
        super().__init__(config)
        self._buy_delta = buy_delta
        self._sell_delta = sell_delta
        self._near_min = near_min_days
        self._near_max = near_max_days
        self._far_min = far_min_days
        self._far_max = far_max_days

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        buy_put = select_put(
            product, date, underlying_price, target_delta=self._buy_delta,
            min_days=self._near_min, max_days=self._near_max,
            loader=loader, um=um, oe=oe,
        )
        sell_put = select_put(
            product, date, underlying_price, target_delta=self._sell_delta,
            min_days=self._far_min, max_days=self._far_max,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if buy_put is not None:
            targets[buy_put.symbol] = 1.0   # 买近月
        if sell_put is not None:
            targets[sell_put.symbol] = -1.0  # 卖远月
        return targets
