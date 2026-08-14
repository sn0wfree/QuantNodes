# coding=utf-8
"""Covered Call: 持有标的 + 卖 OTM call, 收取权利金.

盈利: 标的横盘或小幅上涨, 收取 theta
风险: 标的大幅上涨时 call 被行权, 错过上涨收益
对冲: 已持有标的 ( futures long), 卖 call 收取权利金
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call


@register
class CoveredCall(OptionStrategy):
    """备兑看涨: 期货多头 + 卖 OTM call."""

    name = "covered_call"
    description = "期货多头 + 卖 OTM call, 降低持仓成本"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        target_delta: float = 0.30,
        futures_qty: float = 1.0,
    ):
        super().__init__(config)
        self._target_delta = target_delta
        self._futures_qty = futures_qty

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        call = select_call(
            product, date, underlying_price,
            target_delta=self._target_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        # 期货多头 (持有标的)
        targets[product] = self._futures_qty
        # 卖 call
        if call is not None:
            targets[call.symbol] = -1.0
        return targets
