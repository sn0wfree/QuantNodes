# coding=utf-8
"""v2 Strategy — momentum + stops + VT."""
from __future__ import annotations
from ..common.strategy_engine import BaseStrategy


class V2Strategy(BaseStrategy):
    def __init__(self, pool, rot):
        self.pool, self.rot = pool, rot
        self._pw, self._st = {}, None

    def compute_weights(self, date, pp, nav):
        from ..v2.portfolio_v2 import select_and_weight_v2, apply_stops_v2, equal_weights_v2
        if self._pw:
            self._st = apply_stops_v2(pp, self.pool, self.rot, self._pw, date)
        else:
            self._st = select_and_weight_v2(pp, self.pool, self.rot, date)
        w = self._st.weights
        if not w:
            w = equal_weights_v2(list(pp.columns))
        t = sum(w.values())
        if t > 0:
            w = {k: v / t for k, v in w.items()}
        self._pw = dict(w)
        return w

    def on_risk_check(self, weights, nav, date):
        from ..v2.portfolio_v2 import apply_vol_targeting_v2
        if self.rot.vol_targeting.enabled and self._st and len(nav) > 1:
            apply_vol_targeting_v2(self.rot, nav, date, self._st)
            return self._st.weights
        return weights
