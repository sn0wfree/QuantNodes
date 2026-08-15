# coding=utf-8
"""Covered Put: 期货空头 + 卖 OTM Put.

盈利: 横盘/小跌, 收取 put 权利金
风险: 标的大幅上涨 (期货空头亏损 + put 被行权)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_put


@register
class CoveredPut(OptionStrategy):
    """备兑看跌: 期货空头 + 卖 OTM put."""

    name = "covered_put"
    description = "期货空头 + 卖 OTM put, 降低空头持仓成本"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        target_delta: float = 0.30,
        futures_qty: float = -1.0,
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

        put = select_put(
            product, date, underlying_price,
            target_delta=self._target_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        targets[product] = self._futures_qty  # 期货空头
        if put is not None:
            targets[put.symbol] = -1.0  # 卖 put
        return targets
