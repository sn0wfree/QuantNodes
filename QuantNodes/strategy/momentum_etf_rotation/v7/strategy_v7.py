# coding=utf-8
"""v7 Strategy — TV-PR 因子择时 (expanding_window_tvpr, OOS)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.strategy_engine import BaseStrategy


class V7Strategy(BaseStrategy):
    def __init__(self, cfg=None):
        from ..v7.macro_substrategy_v7_6 import V7_6Config
        self.cfg = cfg or V7_6Config()
        self._loaded = False

    def _load(self):
        if self._loaded: return
        from ..v7.data_loader_v7_6 import load_v7_6_data
        from ..v7.tvpr_estimator import expanding_window_tvpr
        self.X, self.Y, self.codes = load_v7_6_data()
        self.beta = expanding_window_tvpr(self.Y, self.X,
            lambda_tv=self.cfg.lambda_tv, lambda_l1=self.cfg.lambda_l1,
            min_history=self.cfg.min_history, rho=self.cfg.rho,
            max_iter=self.cfg.max_iter, tol=self.cfg.tol)
        self._loaded = True

    def compute_weights(self, date, pp, nav):
        self._load()
        wi = self.Y.index.get_indexer([date], method="ffill")[0]
        if wi < 1: return {}
        b = self.beta.iloc[wi - 1].values
        scores = {}
        for i, c in enumerate(self.codes):
            x = self.X[wi, i, :]
            v = ~np.isnan(x)
            if v.any(): scores[c] = np.dot(x[v], b[v])
        if not scores: return {}
        top = sorted(scores, key=scores.get, reverse=True)[:self.cfg.top_n]
        lb = self.Y.iloc[max(0, wi - 26):wi]
        vols = lb[top].std()
        # 过滤 NaN vol (数据不足的资产)
        valid = vols.dropna()
        if valid.empty: return {}
        vols = valid.clip(lower=self.cfg.vol_floor)
        w = (1.0 / vols)
        w = (w / w.sum()).clip(upper=self.cfg.max_weight)
        return (w / w.sum()).to_dict()
