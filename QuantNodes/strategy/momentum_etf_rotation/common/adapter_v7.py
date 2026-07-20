# coding=utf-8
"""v7 TV-PR 适配器 — 包装 TV-PR 因子择时到统一引擎.

关键修正: 使用 expanding_window_tvpr (OOS, 无前视偏差)
而非 full_sample_tvpr (全量估计, 有前视偏差).

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.adapter_v7 import V7Callbacks
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_engine import run_backtest

    callbacks = V7Callbacks(v7_config)
    result = run_backtest(
        price_panel,  # 周频指数收益面板
        config=BacktestConfig(rebal_freq="W", min_history=52),
        callbacks=callbacks,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import BacktestCallbacks
from .backtest_config import BacktestConfig


class V7Callbacks(BacktestCallbacks):
    """v7.10 TV-PR 适配器.

    核心修正: 用 expanding_window_tvpr 替代 full_sample_tvpr.
    每个 beta_t 仅使用 Y[:t], X[:t] 数据, 无前视偏差.

    数据流:
    1. __init__: 加载因子面板 + 资产收益, 预计算 beta_path (OOS)
    2. compute_weights: 查 beta_path[t-1] → dot(X, beta) → top-N → inv-vol
    3. 引擎: 周频调仓 (rebal_freq="W"), 日频 NAV 自动计算
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: V7_6Config (None = 使用默认)
        """
        from ..v7.macro_substrategy_v7_6 import V7_6Config
        self.cfg = cfg or V7_6Config()

        # 延迟加载
        self._loaded = False
        self.Y = None
        self.X_panel = None
        self.valid_codes = None
        self.beta_path = None

    def _ensure_loaded(self):
        """延迟加载 + OOS beta 估计."""
        if self._loaded:
            return

        from ..v7.data_loader_v7_6 import load_v7_6_data
        from ..v7.tvpr_estimator import expanding_window_tvpr

        X_panel, Y, valid_codes = load_v7_6_data()
        self.Y = Y
        self.X_panel = X_panel
        self.valid_codes = valid_codes

        # 关键修正: expanding_window_tvpr (OOS, 无前视偏差)
        # 每个 beta_t 仅使用 Y[:t], X[:t] 估计
        self.beta_path = expanding_window_tvpr(
            Y, X_panel,
            lambda_tv=self.cfg.lambda_tv,
            lambda_l1=self.cfg.lambda_l1,
            min_history=self.cfg.min_history,
            rho=self.cfg.rho,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
        )

        self._loaded = True

    def compute_signals(self, price_panel, date, state, context):
        self._ensure_loaded()
        return {}

    def select_assets(self, signals, config):
        return []

    def compute_weights(self, selected, price_panel, date, config):
        """TV-PR 信号 → top-N → 逆波动加权."""
        self._ensure_loaded()

        # 找到对应的周频索引
        week_idx = self.Y.index.get_indexer([date], method="ffill")[0]
        if week_idx < 1:
            return {}

        # 信号: dot(X, beta_prev), lag=1 避免前视
        beta_prev = self.beta_path.iloc[week_idx - 1].values
        scores = {}
        for i, code in enumerate(self.valid_codes):
            x = self.X_panel[week_idx, i, :]
            valid_mask = ~np.isnan(x)
            if valid_mask.any():
                scores[code] = np.dot(x[valid_mask], beta_prev[valid_mask])

        if not scores:
            return {}

        # top-N
        chosen = sorted(scores, key=scores.get, reverse=True)[:self.cfg.top_n]

        # 逆波动加权 (26 周窗口)
        lookback = self.Y.iloc[max(0, week_idx - 26):week_idx]
        vols = lookback[chosen].std().clip(lower=self.cfg.vol_floor)
        inv_vol = 1.0 / vols
        weights = (inv_vol / inv_vol.sum()).clip(upper=self.cfg.max_weight)
        weights = weights / weights.sum()

        return weights.to_dict()

    def apply_risk_controls(self, weights, nav_history, date, config):
        """趋势过滤 + 硬止损."""
        if not weights:
            return weights

        # 趋势过滤
        if self.cfg.trend_filter_enabled:
            weights = self._apply_tf(weights, date)

        # 硬止损
        if self.cfg.stop_loss_threshold is not None:
            weights = self._apply_sl(weights, nav_history)

        return weights

    def _apply_tf(self, weights, date):
        """趋势过滤: benchmark 低于 MA 时缩减权益."""
        # 需要 benchmark 日频价格, 延迟到有数据时实现
        return weights

    def _apply_sl(self, weights, nav_history):
        """硬止损: DD 超过阈值时全部转债券."""
        if len(nav_history) < 2:
            return weights
        peak = nav_history.max()
        current = nav_history.iloc[-1]
        dd = current / peak - 1.0
        if dd < self.cfg.stop_loss_threshold:
            # 全部转债券
            return {}
        return weights

    def post_weights(self, weights, config):
        from .backtest_utils import normalize_weights
        return normalize_weights(weights)
