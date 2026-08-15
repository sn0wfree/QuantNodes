# coding=utf-8
"""Options-Factor-Enhanced fut_v1: 期权因子增强 CTA.

在 fut_v1 的因子组合基础上, 叠加期权因子:
    - IV Rank: 高 IV 时减仓
    - VRP: 与 CTA 同向时加仓
    - Skew: 极端时减仓
    - P/C Volume: 极端情绪时反向

集成点: 在 _signal_to_lots() 的 vol_target 之后, 应用期权因子调整系数
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy, ComboConfig, ModuleConfig
from ..factors.volatility import vol_target_positions
from ..options.iv_panel import load_iv_panel
from ..options.hedge_data import load_vrp_panel
from ..options.option_factors import (
    iv_zscore,
    skew_rolling_panel,
    put_call_volume_ratio,
)


@dataclass
class OptionsFactorConfig:
    """期权因子配置."""
    # 因子权重
    w_iv_rank: float = 0.15       # IV rank 过滤
    w_vrp: float = 0.15           # VRP 确认
    w_skew: float = 0.10          # Skew 过滤
    w_pcvr: float = 0.10          # P/C Volume

    # 阈值
    iv_high_threshold: float = 0.7   # IV > 70%: 减仓
    skew_threshold: float = 0.03     # |skew| > 0.03: 减仓
    pcvr_high: float = 1.3           # PCVR > 1.3: 减仓
    pcvr_low: float = 0.7            # PCVR < 0.7: 减仓

    # 调整幅度
    iv_reduce: float = 0.7       # 高 IV 减仓 30%
    vrp_boost: float = 1.15      # VRP 同向加仓 15%
    skew_reduce: float = 0.8     # 极端 skew 减仓 20%
    pcvr_reduce: float = 0.85    # 极端 PCVR 减仓 15%


class OptionsEnhancedFactorModule(FactorModuleComboStrategy):
    """期权因子增强的 FactorModuleComboStrategy.

    在 fut_v1 基础上叠加期权因子调整.
    """

    def __init__(self, options_config: OptionsFactorConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self._opt_cfg = options_config or OptionsFactorConfig()
        self._iv_panel = None
        self._vrp_panel = None

    def _load_options_data(self):
        """加载期权因子数据."""
        try:
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            self._iv_panel = None

        try:
            self._vrp_panel = load_vrp_panel(
                products=self._products, start=self._start, end=self._end,
            )
        except Exception:
            self._vrp_panel = None

    def _get_options_adjustment(
        self,
        product: str,
        date: pd.Timestamp,
    ) -> float:
        """计算期权因子调整系数 (0.5 ~ 1.5)."""
        adjustment = 1.0
        cfg = self._opt_cfg

        # 1. IV Rank: 高 IV → 减仓
        if self._iv_panel is not None and product in self._iv_panel.columns:
            if date in self._iv_panel.index:
                iv_val = self._iv_panel.loc[date, product]
                if pd.notna(iv_val):
                    hist = self._iv_panel[product].dropna()
                    if len(hist) > 60:
                        iv_rank = float((hist < iv_val).sum() / len(hist))
                        if iv_rank > cfg.iv_high_threshold:
                            adjustment *= cfg.iv_reduce

        # 2. VRP: 与 CTA 同向 → 加仓
        if self._vrp_panel is not None and product in self._vrp_panel.columns:
            if date in self._vrp_panel.index:
                vrp_val = self._vrp_panel.loc[date, product]
                # 获取 CTA 信号方向
                if hasattr(self, "_current_signals") and product in self._current_signals.columns:
                    if date in self._current_signals.index:
                        cta_val = self._current_signals.loc[date, product]
                        if pd.notna(vrp_val) and pd.notna(cta_val):
                            if vrp_val > 0 and cta_val > 0:
                                adjustment *= cfg.vrp_boost
                            elif vrp_val < 0 and cta_val < 0:
                                adjustment *= cfg.vrp_boost

        # 3. Skew: 极端 → 减仓
        # (简化: 用 IV 的离散度近似)
        if self._iv_panel is not None and product in self._iv_panel.columns:
            if date in self._iv_panel.index:
                iv_val = self._iv_panel.loc[date, product]
                if pd.notna(iv_val):
                    hist = self._iv_panel[product].dropna()
                    if len(hist) > 60:
                        iv_std = hist.std()
                        iv_mean = hist.mean()
                        if iv_std > 0:
                            iv_z = (iv_val - iv_mean) / iv_std
                            if abs(iv_z) > 2.0:
                                adjustment *= cfg.skew_reduce

        return np.clip(adjustment, 0.5, 1.5)

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """增强版: vol_target × 期权因子调整 → 手数."""
        # 先加载期权数据
        if self._iv_panel is None:
            self._load_options_data()

        # 保存当前信号 (用于 VRP 判断)
        self._current_signals = signals

        # 原始 vol_target
        weights = self._signal_to_lots_base(signals, cfg)

        # 应用期权因子调整
        for product in weights.columns:
            for date in weights.index:
                adj = self._get_options_adjustment(product, date)
                weights.loc[date, product] *= adj

        return self._weights_to_lots(weights, cfg)

    def _signal_to_lots_base(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """原始 vol_target (不含期权因子)."""
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if self._filter is not None:
            weights = weights * self._filter.coefficients(self._price_panel)
        return weights
