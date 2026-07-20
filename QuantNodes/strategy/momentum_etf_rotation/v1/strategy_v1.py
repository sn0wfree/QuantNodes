# coding=utf-8
"""v1 Strategy — hybrid momentum + inverse-vol + stops + VT."""
from __future__ import annotations
import pandas as pd
from ..common.strategy_engine import BaseStrategy


class V1Strategy(BaseStrategy):
    def __init__(self, pool, rot):
        self.pool, self.rot = pool, rot
        self._pw, self._st = {}, None

    def compute_weights(self, date, pp, nav):
        from ..portfolio import select_and_weight, apply_stops, equal_weights
        if self._pw:
            self._st = apply_stops(pp, self.pool, self.rot, self._pw, date)
        else:
            self._st = select_and_weight(pp, self.pool, self.rot, date)
        w = self._st.weights
        if not w:
            w = equal_weights(list(pp.columns))
        t = sum(w.values())
        if t > 0:
            w = {k: v / t for k, v in w.items()}
        self._pw = dict(w)
        return w

    def on_risk_check(self, weights, nav, date):
        from ..portfolio import apply_vol_targeting
        if self.rot.vol_targeting.enabled and self._st and len(nav) > 1:
            apply_vol_targeting(self.rot, nav, date, self._st)
            return self._st.weights
        return weights
