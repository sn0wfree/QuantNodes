# coding=utf-8
"""TV-PR 模块权重组合策略 (策略 v6).

用 TV-PR (Total-Variation Predictive Regression, Cui et al. 2025) 动态决定
7 个模块的权重, 替代固定等权:
    Y (T,N): 品种未来 1 日收益 (横截面)
    X (T,N,K): 7 模块归一化信号 (equal 聚合 ∈ [-1,1])
    β_t (T,K): 时变模块权重 (ADMM, TV 平滑 + λ_l1=0 全维度保留)
    组合信号 = Σ_k β_t,k × X_t,i,k → sign → vol targeting (单一资金池)

无前视: expanding-window OOS 估计 (tvpr_estimator method='expanding'),
每 step=21 天更新一次权重.

用法:
    from QuantNodes.strategy.cta.strategies.tvpr_combo import TVPRComboStrategy

    strat = TVPRComboStrategy(lambda_tv=0.15)
    result = strat.run()
    print(result.metrics)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import CtaBacktestResult
from .factor_modules import (
    EventCalendarModuleStrategy,
    IndustrialChainModuleStrategy,
    MacroModuleStrategy,
    MeanReversionModuleStrategy,
    ModuleConfig,
    TermBasisModuleStrategy,
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
    _ModuleStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    tvpr_estimator,
)

MIN_HISTORY = 252   # TV-PR 最少历史天数 (前 1 年无权重)
STEP = 21           # 权重更新频率 (天, 约月频)


class TVPRComboStrategy(_ModuleStrategy):
    """TV-PR 模块权重组合策略.

    7 个模块信号 (归一化) → TV-PR 时变权重 β_t → 组合信号 = X@β_t → 方向.
    β_t 正负自由: 负 β 表示该模块信号反向预测收益, 自动反用.
    λ_l1=0: 全维度保留 (用户要求每维度都要有).
    """

    def __init__(
        self,
        lambda_tv: float = 0.15,
        lambda_l1: float = 0.0,
        min_history: int = MIN_HISTORY,
        step: int = STEP,
        products: Optional[list[str]] = None,
        config: Optional[ModuleConfig] = None,
        data_loader: Optional[CtaDataLoader] = None,
        start: str = "2018-01-01",
        end: str = "2025-12-31",
    ):
        super().__init__(
            products=products, config=config, data_loader=data_loader,
            start=start, end=end,
        )
        self._lambda_tv = lambda_tv
        self._lambda_l1 = lambda_l1
        self._min_history = min_history
        self._step = step
        cfg = self._config
        self._modules = [
            TrendMomentumModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            MeanReversionModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            VolumeFlowModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            EventCalendarModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            TermBasisModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            MacroModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            IndustrialChainModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
        ]
        self.module_signals: dict[str, pd.DataFrame] = {}
        self.beta_path: pd.DataFrame | None = None

    def load_panel(self) -> pd.DataFrame:
        panel = super().load_panel()
        for m in self._modules:
            m._price_panel = panel
            m._ohlc_map = self._ohlc_map
            m._vols = self._vols
            m._ois = self._ois
        return panel

    def _module_panel(self) -> pd.DataFrame:
        """7 模块信号 → (T, 7) 面板 (逐品种展平均值, 归一化)."""
        if not self.module_signals:
            self._collect_modules()
        return pd.DataFrame({
            name: s.to_numpy().ravel()
            for name, s in self.module_signals.items()
        })

    def _collect_modules(self) -> None:
        for m in self._modules:
            self.module_signals[m.__class__.__name__] = m._signals_raw()

    def _estimate_beta(self) -> pd.DataFrame:
        """TV-PR expanding 估计 β_t (T, 7).

        X (T,N,7): 模块信号面板; Y (T,N): 品种未来 1 日收益.
        返回前 min_history 行为 0 (expanding_window_tvpr 内处理).
        """
        if self._price_panel is None:
            self.load_panel()
        self._collect_modules()
        names = list(self.module_signals)
        cols = list(self._price_panel.columns)
        X = np.stack([
            self.module_signals[n].reindex(columns=cols).values
            for n in names
        ], axis=-1)                       # (T,N,7)
        Y = self._price_panel.pct_change(fill_method=None).shift(-1)   # (T,N) 未来收益
        beta = tvpr_estimator(
            Y, X,
            lambda_tv=self._lambda_tv,
            lambda_l1=self._lambda_l1,
            method="expanding",
            min_history=self._min_history,
            step=self._step,
        )
        beta.columns = names
        self.beta_path = beta
        return beta

    def _signals_raw(self) -> pd.DataFrame:
        """组合信号 = Σ_k β_t,k × X_t,i,k (T,N)."""
        beta = self._estimate_beta()
        names = list(self.module_signals)
        cols = list(self._price_panel.columns)
        X = np.stack([
            self.module_signals[n].reindex(columns=cols).values
            for n in names
        ], axis=-1)
        combo = np.einsum("tni,ti->tn", X, beta.values)
        return pd.DataFrame(combo, index=self._price_panel.index, columns=cols)

    def _signals(self) -> pd.DataFrame:
        return np.sign(self._signals_raw())

    def run(self) -> CtaBacktestResult:
        """运行完整回测 (方向变化才调仓)."""
        self.load_panel()
        return self._run_signal(self._signals())

    def beta_summary(self) -> pd.DataFrame:
        """β_t 均值/非零率/波动 (权重诊断)."""
        beta = self._estimate_beta()
        summary = pd.DataFrame({
            "mean": beta.mean(),
            "std": beta.std(),
            "nonzero": (beta.abs() > 1e-9).mean(),
            "min": beta.min(),
            "max": beta.max(),
        })
        return summary
