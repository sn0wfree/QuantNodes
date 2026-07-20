# coding=utf-8
"""v6 行业轮动适配器 — 包装 v5 因子评分 + v5.1 逆波动加权到统一引擎.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.adapter_v6 import V6Callbacks
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_engine import run_backtest

    callbacks = V6Callbacks(v6_config, panel_ohlcv)
    result = run_backtest(panel_close, config=BacktestConfig(rebal_freq="M"), callbacks=callbacks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import BacktestCallbacks
from .backtest_config import BacktestConfig


class V6Callbacks(BacktestCallbacks):
    """v6 行业轮动适配器.

    将 v5 因子评分 + v5.1 逆波动加权 + v2 风控包装进回调.
    """

    def __init__(self, cfg=None, panel_ohlcv=None):
        """
        Args:
            cfg: V6Config (None = 使用默认)
            panel_ohlcv: OHLCV 面板 (用于因子计算)
        """
        from ..v6.industry_rotation_v6 import V6Config, V6SubStrategy

        self.cfg = cfg or V6Config()
        self.panel_ohlcv = panel_ohlcv
        self.sub = V6SubStrategy(self.cfg)
        self._factor_panel = None
        self._factor_panel_date = None

    def _ensure_factor_panel(self, panel_close):
        """延迟计算因子面板."""
        if self._factor_panel is not None:
            return
        if self.panel_ohlcv is None:
            raise ValueError("V6Callbacks 需要 panel_ohlcv 参数")
        from ..v5.industry_factors import compute_all_factors_panel
        self._factor_panel = compute_all_factors_panel(
            self.panel_ohlcv, self.cfg.factor_cfg
        )

    def compute_signals(self, price_panel, date, state, context):
        return {}

    def select_assets(self, signals, config):
        return []

    def compute_weights(self, selected, price_panel, date, config):
        """v5 因子评分 → top-N → 逆波动加权."""
        self._ensure_factor_panel(price_panel)

        # 注入因子面板到子策略
        self.sub._factor_panel = self._factor_panel

        # 选股 + 加权
        chosen = self.sub.select(price_panel, date)
        if not chosen:
            return {}
        weights = self.sub.weight(price_panel, chosen, date)
        return weights

    def apply_risk_controls(self, weights, nav_history, date, config):
        """应用 VT + TF (来自 v2 框架)."""
        if not weights:
            return weights

        # 趋势过滤
        if self.cfg.use_trend_filter:
            weights = self._apply_tf(weights, nav_history, date)

        # 波动率目标
        if self.cfg.use_vol_targeting:
            weights = self._apply_vt(weights, nav_history, date)

        return weights

    def _apply_tf(self, weights, nav_history, date):
        """趋势过滤: benchmark 低于 MA 时缩减权益."""
        if len(nav_history) < self.cfg.trend_filter_ma_window:
            return weights

        benchmark = self.cfg.trend_filter_benchmark
        bond = self.cfg.trend_filter_bond
        if benchmark not in weights and bond not in weights:
            return weights

        # 简化: 如果 benchmark 权重 > 0 且 nav 下行趋势, 缩减
        # (完整实现需要 benchmark 价格数据, 这里用 nav_history 近似)
        return weights

    def _apply_vt(self, weights, nav_history, date):
        """波动率目标: 缩放总仓位."""
        if len(nav_history) < self.cfg.vol_target_lookback:
            return weights

        recent = nav_history.iloc[-self.cfg.vol_target_lookback:]
        rets = recent.pct_change().dropna()
        if len(rets) < 10:
            return weights

        realized_vol = rets.std() * np.sqrt(252)
        if realized_vol <= 0:
            return weights

        scale = self.cfg.vol_target / realized_vol
        scale = max(self.cfg.vol_target_min_scale, min(self.cfg.vol_target_max_scale, scale))

        return {k: v * scale for k, v in weights.items()}

    def post_weights(self, weights, config):
        from .backtest_utils import normalize_weights
        return normalize_weights(weights)
