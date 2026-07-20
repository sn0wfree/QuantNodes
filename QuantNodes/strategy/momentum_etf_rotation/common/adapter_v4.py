# coding=utf-8
"""v4 多策略适配器 — 包装 style + smart_beta + factor_timing 到统一引擎.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.adapter_v4 import V4Callbacks
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_engine import run_backtest

    callbacks = V4Callbacks(v4_config)
    result = run_backtest(panel, config=BacktestConfig(rebal_freq="M"), callbacks=callbacks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import BacktestCallbacks
from .backtest_config import BacktestConfig


class V4Callbacks(BacktestCallbacks):
    """v4 多策略适配器.

    将 style_rotation + smart_beta + factor_timing 包装进 compute_weights.
    """

    def __init__(self, cfg=None, hmm_regime_series=None):
        """
        Args:
            cfg: V4Config (None = 使用默认)
            hmm_regime_series: HMM regime 序列 (可选, v4E/v4F)
        """
        from ..v4.multi_strategy_v4 import V4Config
        from ..v4.style_rotation_v4 import StyleRotationSubStrategy
        from ..v4.smart_beta_v4 import SmartBetaSubStrategy
        from ..v4.factor_timing_v4 import (
            compute_factor_weights,
            compute_strategy_weights,
            backtest_factor_timing,
        )

        self.cfg = cfg or V4Config()
        self.hmm_regime_series = hmm_regime_series

        # 子策略
        self.style_sub = None
        if self.cfg.style_enabled:
            self.style_sub = StyleRotationSubStrategy(self.cfg.style)

        self.sb_sub = None
        if self.cfg.smart_beta_enabled:
            self.sb_sub = SmartBetaSubStrategy(self.cfg.smart_beta)

        # 因子择时 IC 历史 (预计算)
        self._ic_history = None
        if self.cfg.factor_timing_enabled:
            # IC 需要从 panel 预计算, 延迟到首次调用
            pass

    def _ensure_ic_history(self, panel):
        """延迟计算 IC 历史."""
        if self._ic_history is not None:
            return
        if not self.cfg.factor_timing_enabled:
            return
        from ..v4.factor_timing_v4 import backtest_factor_timing
        try:
            self._ic_history = backtest_factor_timing(panel, self.cfg.factor_timing)
        except Exception:
            self._ic_history = pd.DataFrame()

    def compute_signals(self, price_panel, date, state, context):
        return {}

    def select_assets(self, signals, config):
        return []

    def compute_weights(self, selected, price_panel, date, config):
        """运行 style + smart_beta 子策略 + 因子择时合并."""
        from ..v4.sub_strategy_v4 import SubStrategyResult
        from ..v4.multi_strategy_v4 import _combine_sub_results

        self._ensure_ic_history(price_panel)

        sub_results = []
        if self.style_sub is not None:
            r = self.style_sub.run_step(price_panel, date)
            sub_results.append(r)

        if self.sb_sub is not None:
            r = self.sb_sub.run_step(price_panel, date)
            sub_results.append(r)

        if not sub_results:
            return {}

        # 子策略权重 (静态或动态)
        sub_weights = {
            "style_rotation": self.cfg.style_weight,
            "smart_beta": self.cfg.smart_beta_weight,
        }

        # 因子择时: 动态调整子策略权重
        if (self.cfg.factor_timing_enabled
                and self._ic_history is not None
                and not self._ic_history.empty):
            from ..v4.factor_timing_v4 import compute_factor_weights, compute_strategy_weights

            idx = self._ic_history.index.get_indexer([date], method="ffill")[0]
            if idx >= 0:
                ic_dict = self._ic_history.iloc[idx].to_dict()
                f_w = compute_factor_weights(
                    pd.DataFrame([ic_dict], index=[date]),
                    self.cfg.factor_timing,
                )

                # HMM regime 调整
                if self.hmm_regime_series is not None:
                    current_regime = -1
                    if date in self.hmm_regime_series.index:
                        current_regime = int(self.hmm_regime_series.loc[date])
                    elif len(self.hmm_regime_series) > 0:
                        idx_r = self.hmm_regime_series.index.get_indexer(
                            [date], method="ffill"
                        )[0]
                        if idx_r >= 0:
                            current_regime = int(self.hmm_regime_series.iloc[idx_r])

                    if current_regime >= 0:
                        from ..v4.regime_detector_v4 import get_regime_factor_weight
                        adjusted = {}
                        for f, w in f_w.items():
                            regime_w = get_regime_factor_weight(current_regime, f)
                            adjusted[f] = w * regime_w
                        total_adj = sum(adjusted.values())
                        if total_adj > 0:
                            adjusted = {k: v / total_adj for k, v in adjusted.items()}
                        f_w = adjusted

                s_w = compute_strategy_weights(
                    f_w, self.cfg.factor_timing.factor_to_strategy,
                )
                if s_w:
                    sub_weights = s_w
                    total = sum(sub_weights.values())
                    if total > 0:
                        sub_weights = {k: v / total for k, v in sub_weights.items()}

        # 合并
        combined = _combine_sub_results(sub_results, sub_weights)
        return combined

    def apply_risk_controls(self, weights, nav_history, date, config):
        return weights

    def post_weights(self, weights, config):
        from .backtest_utils import apply_max_weight, normalize_weights
        weights = apply_max_weight(weights, self.cfg.max_weight)
        return normalize_weights(weights)
