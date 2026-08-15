# coding=utf-8
"""Iron Butterfly: 卖 ATM call + 卖 ATM put + 买 OTM call + 买 OTM put.

盈利: 标的到期在 ATM 附近
风险: 封顶 (最大亏损 = 宽度 - 净权利金)
优势: 比 strangle 风险更可控
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class IronButterfly(OptionStrategy):
    """铁蝶式: 卖 ATM + 买 OTM wings."""

    name = "iron_butterfly"
    description = "卖 ATM call+put + 买 OTM wings, 风险封顶"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        wing_width_pct: float = 0.03,
    ):
        super().__init__(config)
        self._wing_width_pct = wing_width_pct

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        width = underlying_price * self._wing_width_pct

        # ATM: 卖
        sell_call = select_call(
            product, date, underlying_price, target_delta=0.50,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        sell_put = select_put(
            product, date, underlying_price, target_delta=0.50,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        # OTM wings: 买
        buy_call = select_call(
            product, date, underlying_price, target_delta=0.15,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        buy_put = select_put(
            product, date, underlying_price, target_delta=0.15,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if sell_call is not None:
            targets[sell_call.symbol] = -1.0
        if sell_put is not None:
            targets[sell_put.symbol] = -1.0
        if buy_call is not None and buy_call.strike > (sell_call.strike if sell_call else 0):
            targets[buy_call.symbol] = 1.0
        if buy_put is not None and buy_put.strike < (sell_put.strike if sell_put else 0):
            targets[buy_put.symbol] = 1.0
        return targets
