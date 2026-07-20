# coding=utf-8
"""v4 Strategy — style rotation + smart beta + factor timing."""
from __future__ import annotations
import pandas as pd
from ..common.strategy_engine import BaseStrategy


class V4Strategy(BaseStrategy):
    def __init__(self, cfg=None, hmm_regime_series=None):
        from ..v4.multi_strategy_v4 import V4Config
        from ..v4.style_rotation_v4 import StyleRotationSubStrategy
        from ..v4.smart_beta_v4 import SmartBetaSubStrategy

        self.cfg = cfg or V4Config()
        self.hmm = hmm_regime_series
        self.style = StyleRotationSubStrategy(self.cfg.style) if self.cfg.style_enabled else None
        self.sbeta = SmartBetaSubStrategy(self.cfg.smart_beta) if self.cfg.smart_beta_enabled else None
        self._ic = None

    def compute_weights(self, date, pp, nav):
        from ..v4.multi_strategy_v4 import _combine_sub_results
        subs = []
        if self.style: subs.append(self.style.run_step(pp, date))
        if self.sbeta: subs.append(self.sbeta.run_step(pp, date))
        if not subs: return {}

        sw = {"style_rotation": self.cfg.style_weight, "smart_beta": self.cfg.smart_beta_weight}

        if self.cfg.factor_timing_enabled and self._ic is not None and not self._ic.empty:
            from ..v4.factor_timing_v4 import compute_factor_weights, compute_strategy_weights
            idx = self._ic.index.get_indexer([date], method="ffill")[0]
            if idx >= 0:
                fw = compute_factor_weights(pd.DataFrame([self._ic.iloc[idx].to_dict()], index=[date]), self.cfg.factor_timing)
                s = compute_strategy_weights(fw, self.cfg.factor_timing.factor_to_strategy)
                if s:
                    t = sum(s.values())
                    sw = {k: v / t for k, v in s.items()} if t > 0 else sw

        return _combine_sub_results(subs, sw)

    def on_risk_check(self, weights, nav, date):
        from ..common.backtest_utils import apply_max_weight, normalize_weights
        return normalize_weights(apply_max_weight(weights, self.cfg.max_weight))
