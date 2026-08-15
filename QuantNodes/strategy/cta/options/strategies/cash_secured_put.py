# coding=utf-8
"""Cash-Secured Put: 卖 OTM Put + 全额保证金.

盈利: 标的横盘/上涨, 收取 put 权利金
风险: 标的下跌时被迫接货 (按行权价买入)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_put


@register
class CashSecuredPut(OptionStrategy):
    """现金担保卖 Put: 卖 OTM put, 愿意接货."""

    name = "cash_secured_put"
    description = "卖 OTM put + 全额保证金, 愿意接货时收取权利金"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        target_delta: float = 0.30,
    ):
        super().__init__(config)
        self._target_delta = target_delta

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

        if put is not None:
            return {put.symbol: -1.0}  # 卖 put
        return {}
