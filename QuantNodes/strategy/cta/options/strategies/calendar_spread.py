# coding=utf-8
"""Calendar Spread: 卖近月 + 买远月, 赚时间价值差.

盈利: 近月 theta 衰减 > 远月, IV 不变或下降
风险: 远月权利金 > 近月 (净支出)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_put


@register
class CalendarSpread(OptionStrategy):
    """日历价差: 卖近月 put + 买远月 put."""

    name = "calendar_spread"
    description = "卖近月 + 买远月, 赚 theta 差"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        target_delta: float = 0.25,
        near_min_days: int = 15,
        near_max_days: int = 35,
        far_min_days: int = 45,
        far_max_days: int = 75,
    ):
        super().__init__(config)
        self._target_delta = target_delta
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

        # 近月 put (卖)
        near = select_put(
            product, date, underlying_price,
            target_delta=self._target_delta,
            min_days=self._near_min, max_days=self._near_max,
            loader=loader, um=um, oe=oe,
        )
        # 远月 put (买)
        far = select_put(
            product, date, underlying_price,
            target_delta=self._target_delta,
            min_days=self._far_min, max_days=self._far_max,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if near is not None:
            targets[near.symbol] = -1.0  # 卖近月
        if far is not None:
            targets[far.symbol] = 1.0   # 买远月
        return targets
