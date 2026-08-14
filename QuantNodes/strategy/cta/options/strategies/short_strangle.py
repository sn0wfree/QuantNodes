# coding=utf-8
"""Short Strangle: 卖 OTM call + OTM put, 收取权利金.

盈利: realized vol < IV, 标的在两腿之间到期
风险: 单边突破 (亏损可能很大)
对冲: 可选 delta hedge (卖出后 delta 为负, 用 futures 对冲)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class ShortStrangle(OptionStrategy):
    """卖宽跨式: 卖 OTM call + OTM put."""

    name = "short_strangle"
    description = "卖 OTM call + OTM put, 收取 theta, 风险无限"

    def __init__(self, config: StrategyConfig | None = None, call_delta: float = 0.25, put_delta: float = 0.25):
        super().__init__(config)
        self._call_delta = call_delta
        self._put_delta = put_delta

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        if existing_holding is not None:
            return {}

        call = select_call(
            product, date, underlying_price,
            target_delta=self._call_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        put = select_put(
            product, date, underlying_price,
            target_delta=self._put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )

        targets = {}
        if call is not None:
            targets[call.symbol] = -1.0  # 卖
        if put is not None:
            targets[put.symbol] = -1.0  # 卖
        return targets
