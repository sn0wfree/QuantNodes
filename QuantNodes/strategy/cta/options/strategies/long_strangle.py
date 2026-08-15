# coding=utf-8
"""Long Strangle: 买 OTM call + 买 OTM put.

盈利: 标的大幅波动 (方向不限)
风险: 权利金归零 (两腿都 OTM)
优势: 比 straddle 便宜 (两腿都 OTM)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put


@register
class LongStrangle(OptionStrategy):
    """长宽跨式: 买 OTM call + 买 OTM put."""

    name = "long_strangle"
    description = "买 OTM call + 买 OTM put, 赌大幅波动"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        call_delta: float = 0.25,
        put_delta: float = 0.25,
    ):
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
            targets[call.symbol] = 1.0  # 买 call
        if put is not None:
            targets[put.symbol] = 1.0   # 买 put
        return targets
