# coding=utf-8
"""v3 多策略适配器 — 包装 3 个子策略到统一引擎.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.adapter_v3 import V3Callbacks
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_engine import run_backtest
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_config import BacktestConfig

    callbacks = V3Callbacks(pool, v3_config)
    result = run_backtest(etf_nav_norm, config=BacktestConfig(rebal_freq="M"), callbacks=callbacks)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .backtest_engine import BacktestCallbacks
from .backtest_config import BacktestConfig


class V3Callbacks(BacktestCallbacks):
    """v3 多策略适配器.

    将 momentum + reversion + industry_rotation 3 个子策略
    包装进 compute_weights 回调, 复用统一引擎的 NAV 计算和指标.
    """

    def __init__(
        self,
        pool,
        cfg=None,
    ):
        """
        Args:
            pool: ETFPool 实例
            cfg: MultiStrategyConfig (None = 使用默认)
        """
        from ..v3.multi_strategy_v3 import (
            MultiStrategyConfig,
            _MomentumShim,
            _simple_momentum_select,
            _simple_momentum_weight,
        )
        from ..v3.sub_strategy_v3 import SubStrategyConfig, SubStrategy
        from ..v3.reversion_v3 import ReversionSubStrategy, ReversionConfig
        from ..v3.industry_rotation_v3 import (
            IndustryRotationSubStrategy,
            IndustryRotationConfig,
        )

        self.pool = pool
        self.cfg = cfg or MultiStrategyConfig()

        # 动量子策略 (shim)
        self.mom_shim = None
        if self.cfg.momentum_enabled:
            class _MomShim(SubStrategy):
                def select(self, nav_df, as_of):
                    return _simple_momentum_select(
                        nav_df, as_of, pool,
                        lookback=self.config.min_history,
                        top_n=self.config.top_n,
                    )
                def weight(self, nav_df, codes, as_of):
                    return _simple_momentum_weight(nav_df, codes, as_of)
                def run_step(self, nav_df, as_of):
                    codes = self.select(nav_df, as_of)
                    weights = self.weight(nav_df, codes, as_of)
                    from ..v3.sub_strategy_v3 import SubStrategyResult
                    return SubStrategyResult(
                        date=as_of, chosen=codes, weights=weights,
                        meta={"strategy": "momentum"},
                    )

            self.mom_shim = _MomShim(
                SubStrategyConfig(
                    name="momentum",
                    top_n=self.cfg.momentum_top_n,
                    min_history=self.cfg.momentum_lookback,
                ),
                pool,
            )

        # 反转子策略
        self.rev_sub = None
        if self.cfg.reversion_enabled:
            self.rev_sub = ReversionSubStrategy(self.cfg.reversion, pool)

        # 行业轮动子策略
        self.ind_sub = None
        if self.cfg.industry_rotation_enabled:
            self.ind_sub = IndustryRotationSubStrategy(
                self.cfg.industry_rotation, pool
            )

    def compute_signals(self, price_panel, date, state, context):
        return {}

    def select_assets(self, signals, config):
        return []

    def compute_weights(self, selected, price_panel, date, config):
        """运行 3 个子策略 + 合并权重."""
        from ..v3.sub_strategy_v3 import SubStrategyResult
        from ..v3.sub_weighting_v3 import (
            combine_sub_results,
            sub_weights_from_results_safe,
        )

        sub_results = []

        if self.mom_shim is not None:
            r = self.mom_shim.run_step(price_panel, date)
            sub_results.append(r)

        if self.rev_sub is not None:
            r = self.rev_sub.run_step(price_panel, date)
            sub_results.append(r)

        if self.ind_sub is not None:
            r = self.ind_sub.run_step(price_panel, date)
            sub_results.append(r)

        if not sub_results:
            return {}

        # 子策略间权重
        sub_w = sub_weights_from_results_safe(
            sub_results, self.cfg.weight_method
        )

        # 合并
        combined = combine_sub_results(
            sub_results, sub_w, pool_codes=set(self.pool.codes)
        )

        return combined

    def apply_risk_controls(self, weights, nav_history, date, config):
        return weights

    def post_weights(self, weights, config):
        from .backtest_utils import apply_max_weight, normalize_weights
        weights = apply_max_weight(weights, self.cfg.max_weight)
        return normalize_weights(weights)
