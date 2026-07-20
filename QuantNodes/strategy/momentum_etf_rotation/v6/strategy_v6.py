# coding=utf-8
"""v6 Strategy — v5 factor scoring + v5.1 inverse-vol."""
from __future__ import annotations
import pandas as pd
from ..common.strategy_engine import BaseStrategy


class V6Strategy(BaseStrategy):
    def __init__(self, cfg=None, panel_ohlcv=None):
        from ..v6.industry_rotation_v6 import V6Config, V6SubStrategy
        self.cfg = cfg or V6Config()
        self.ohlcv = panel_ohlcv
        self.sub = V6SubStrategy(self.cfg)
        self._fp = None

    def compute_weights(self, date, pp, nav):
        if self._fp is not None:
            self.sub._factor_panel = self._fp
        chosen = self.sub.select(pp, date)
        if not chosen: return {}
        return self.sub.weight(pp, chosen, date)

    def on_risk_check(self, weights, nav, date):
        if not weights or not self.cfg.use_vol_targeting: return weights
        if len(nav) < self.cfg.vol_target_lookback: return weights
        rets = nav.iloc[-self.cfg.vol_target_lookback:].pct_change().dropna()
        if len(rets) < 10: return weights
        vol = rets.std() * 252 ** 0.5
        if vol <= 0: return weights
        s = max(self.cfg.vol_target_min_scale,
                min(self.cfg.vol_target_max_scale, self.cfg.vol_target / vol))
        return {k: v * s for k, v in weights.items()}
