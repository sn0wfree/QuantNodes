# coding=utf-8
"""Butterfly Spread: 买1低 + 卖2中 + 买1高.

盈利: 标的到期在中间 strike 附近
风险: 封顶 (最大亏损 = 净权利金)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class LongButterfly(OptionStrategy):
    """长蝶式: 买1低 strike + 卖2中间 strike + 买1高 strike."""

    name = "long_butterfly"
    description = "买1低 + 卖2中 + 买1高, 赌标的在中间到期"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        option_type: str = "call",
        wing_width_pct: float = 0.03,
    ):
        super().__init__(config)
        self._option_type = option_type
        self._wing_width_pct = wing_width_pct

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        # 选3个 strike: ATM-宽, ATM, ATM+宽
        width = underlying_price * self._wing_width_pct
        low_strike = underlying_price - width
        high_strike = underlying_price + width

        select_fn = select_call if self._option_type == "call" else select_put

        # 低 strike (买)
        low = select_fn(
            product, date, underlying_price,
            target_delta=0.5,  # ATM
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        # 中间 strike (卖)
        mid = select_fn(
            product, date, underlying_price,
            target_delta=0.5,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        # 高 strike (买)
        high = select_fn(
            product, date, underlying_price,
            target_delta=0.5,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        if low is None or mid is None or high is None:
            return {}

        # 确保 low.strike < mid.strike < high.strike
        strikes = sorted([low.strike, mid.strike, high.strike])
        if len(set(strikes)) < 3:
            return {}

        targets = {}
        targets[low.symbol] = 1.0      # 买低
        targets[mid.symbol] = -2.0     # 卖2中间
        targets[high.symbol] = 1.0     # 买高
        return targets
