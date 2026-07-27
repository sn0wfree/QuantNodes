# coding=utf-8
"""v3 Strategy — momentum + reversion + industry rotation."""
from __future__ import annotations
from ..common.strategy_engine import BaseStrategy


class V3Strategy(BaseStrategy):
    def __init__(self, pool, cfg):
        from ..v3.multi_strategy_v3 import (
            MultiStrategyConfig, _simple_momentum_select, _simple_momentum_weight,
        )
        from ..v3.sub_strategy_v3 import SubStrategyConfig, SubStrategy, SubStrategyResult
        from ..v3.reversion_v3 import ReversionSubStrategy
        from ..v3.industry_rotation_v3 import IndustryRotationSubStrategy

        self.pool, self.cfg = pool, cfg or MultiStrategyConfig()

        self.mom = None
        if self.cfg.momentum_enabled:
            class _M(SubStrategy):
                def select(self, nav_df, as_of):
                    return _simple_momentum_select(nav_df, as_of, pool,
                        lookback=self.config.min_history, top_n=self.config.top_n)
                def weight(self, nav_df, codes, as_of):
                    return _simple_momentum_weight(nav_df, codes, as_of)
                def run_step(self, nav_df, as_of):
                    c, w = self.select(nav_df, as_of), self.weight(nav_df, self.select(nav_df, as_of), as_of)
                    return SubStrategyResult(date=as_of, chosen=c, weights=w, meta={"strategy": "momentum"})
            self.mom = _M(SubStrategyConfig(name="momentum", top_n=cfg.momentum_top_n,
                                            min_history=cfg.momentum_lookback), pool)

        self.rev = ReversionSubStrategy(cfg.reversion, pool) if cfg.reversion_enabled else None
        self.ind = IndustryRotationSubStrategy(cfg.industry_rotation, pool) if cfg.industry_rotation_enabled else None

    def compute_weights(self, date, pp, nav):
        from ..v3.sub_weighting_v3 import combine_sub_results, sub_weights_from_results
        subs = []
        if self.mom: subs.append(self.mom.run_step(pp, date))
        if self.rev: subs.append(self.rev.run_step(pp, date))
        if self.ind: subs.append(self.ind.run_step(pp, date))
        if not subs: return {}
        sw = sub_weights_from_results(subs, self.cfg.weight_method)
        return combine_sub_results(subs, sw, pool_codes=set(self.pool.codes))

    def on_risk_check(self, weights, nav, date):
        from ..common.backtest_utils import apply_max_weight, normalize_weights
        return normalize_weights(apply_max_weight(weights, self.cfg.max_weight))
