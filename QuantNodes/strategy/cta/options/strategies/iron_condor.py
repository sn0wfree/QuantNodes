# coding=utf-8
"""Iron Condor: 卖 OTM put + 买更 OTM put + 卖 OTM call + 买更 OTM call.

盈利: 标的在两腿之间, realized vol < IV
风险: 封顶 (最大亏损 = 宽度 - 净权利金)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class IronCondor(OptionStrategy):
    """铁秃鹰: 两腿价差组合, 风险封顶."""

    name = "iron_condor"
    description = "卖 put spread + sell call spread, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        sell_call_delta: float = 0.25,
        buy_call_delta: float = 0.10,
        sell_put_delta: float = 0.25,
        buy_put_delta: float = 0.10,
    ):
        super().__init__(config)
        self._sell_call_delta = sell_call_delta
        self._buy_call_delta = buy_call_delta
        self._sell_put_delta = sell_put_delta
        self._buy_put_delta = buy_put_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        # Call spread: 卖低 delta + 买高 delta (更 OTM)
        sell_call = select_call(
            product, date, underlying_price,
            target_delta=self._sell_call_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        buy_call = select_call(
            product, date, underlying_price,
            target_delta=self._buy_call_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        # Put spread: 卖高 delta + 买低 delta (更 OTM)
        sell_put = select_put(
            product, date, underlying_price,
            target_delta=self._sell_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        buy_put = select_put(
            product, date, underlying_price,
            target_delta=self._buy_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if sell_call is not None:
            targets[sell_call.symbol] = -1.0
        if buy_call is not None and buy_call.strike > (sell_call.strike if sell_call else 0):
            targets[buy_call.symbol] = 1.0
        if sell_put is not None:
            targets[sell_put.symbol] = -1.0
        if buy_put is not None and buy_put.strike < (sell_put.strike if sell_put else float("inf")):
            targets[buy_put.symbol] = 1.0
        return targets
