# coding=utf-8
"""Options-Enhanced CTA: 期权因子增强 CTA 期货策略.

核心思路: 不交易期权, 用期权数据增强 CTA 期货信号.

增强逻辑:
    1. IV Rank 过滤: 高 IV 时减仓 (vol clustering 风险)
    2. VRP 确认: VRP 与 CTA 同向时加仓 (期权市场确认)
    3. Skew 过滤: 极端 skew 时减仓 (尾部风险)
    4. P/C Volume: 极端情绪时反向调整

收益来源: 纯 CTA 期货 alpha + 期权因子信息优势
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_risk import PositionLimitRisk
from ..options.iv_panel import load_iv_panel
from ..options.hedge_data import load_vrp_panel
from ..options.option_factors import (
    iv_zscore,
    skew_rolling_panel,
    put_call_volume_ratio,
    compute_gex,
    gex_zscore,
)


@dataclass
class OptionsEnhancedCTAConfig:
    """期权增强 CTA 配置."""
    # 品种
    products: list[str] = field(default_factory=lambda: ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR"])

    # CTA 参数
    initial_cash: float = 10_000_000.0
    target_vol: float = 0.15
    max_leverage: float = 3.0
    vol_lookback: int = 20

    # 期权因子权重
    w_iv_rank: float = 0.15       # IV rank 过滤权重
    w_vrp: float = 0.15           # VRP 确认权重
    w_skew: float = 0.10          # Skew 过滤权重
    w_pcvr: float = 0.10          # P/C Volume 权重
    w_cta: float = 0.50           # CTA 基础信号权重

    # 过滤阈值
    iv_high_threshold: float = 0.7   # IV > 70%: 减仓
    skew_extreme_threshold: float = 1.5  # |skew_z| > 1.5: 减仓
    pcvr_extreme_threshold: float = 1.3  # PCVR > 1.3: 反向

    # 回测
    start: str = "2023-01-01"
    end: str = "2025-12-31"


class OptionsEnhancedCTA(CtaStrategy):
    """期权因子增强 CTA 策略.

    信号 = CTA 基础信号 × 期权因子调整系数
    """

    def __init__(
        self,
        price_panel: pd.DataFrame,
        config: OptionsEnhancedCTAConfig,
        iv_panel: pd.DataFrame | None = None,
        vrp_panel: pd.DataFrame | None = None,
        skew_panel: pd.DataFrame | None = None,
        pcvr_series: pd.Series | None = None,
    ):
        self._price_panel = price_panel
        self._cfg = config
        self._iv_panel = iv_panel
        self._vrp_panel = vrp_panel
        self._skew_panel = skew_panel
        self._pcvr_series = pcvr_series

        # 预计算 CTA 信号
        self._cta_signals = self._compute_cta_signals()

    def _compute_cta_signals(self) -> pd.DataFrame:
        """计算 CTA 基础信号 (多因子投票)."""
        signals = pd.DataFrame(0.0, index=self._price_panel.index, columns=self._price_panel.columns)

        for product in self._price_panel.columns:
            series = self._price_panel[product]
            ma5 = series.rolling(5).mean()
            ma20 = series.rolling(20).mean()
            ma60 = series.rolling(60).mean()
            mom5 = series.pct_change(5)
            mom20 = series.pct_change(20)
            mom60 = series.pct_change(60)

            # 多因子投票
            sig = (
                (ma5 > ma20).astype(float) * 0.25
                + (ma20 > ma60).astype(float) * 0.25
                + (mom5 > 0).astype(float) * 0.15
                + (mom20 > 0).astype(float) * 0.20
                + (mom60 > 0).astype(float) * 0.15
            ) * 2 - 1  # 映射到 [-1, 1]

            signals[product] = sig

        return signals

    def _get_options_adjustment(
        self,
        product: str,
        date: pd.Timestamp,
    ) -> float:
        """计算期权因子调整系数.

        返回: 0.5 ~ 1.5 的乘数
            < 1.0: 减仓 (期权市场警告)
            > 1.0: 加仓 (期权市场确认)
            = 1.0: 无调整
        """
        adjustment = 1.0

        # 1. IV Rank 过滤: 高 IV → 减仓
        if self._iv_panel is not None and product in self._iv_panel.columns:
            if date in self._iv_panel.index:
                iv_val = self._iv_panel.loc[date, product]
                if pd.notna(iv_val):
                    hist = self._iv_panel[product].dropna()
                    if len(hist) > 60:
                        iv_rank = float((hist < iv_val).sum() / len(hist))
                        if iv_rank > self._cfg.iv_high_threshold:
                            adjustment *= 0.7  # 高 IV 减仓 30%

        # 2. VRP 确认: VRP 与 CTA 同向 → 加仓
        if self._vrp_panel is not None and product in self._vrp_panel.columns:
            if date in self._vrp_panel.index:
                vrp_val = self._vrp_panel.loc[date, product]
                cta_val = self._cta_signals.loc[date, product] if date in self._cta_signals.index else 0
                if pd.notna(vrp_val) and not np.isnan(cta_val):
                    # VRP > 0 且 CTA > 0: 期权市场确认看涨
                    if vrp_val > 0 and cta_val > 0:
                        adjustment *= 1.15  # 加仓 15%
                    # VRP < 0 且 CTA < 0: 期权市场确认看跌
                    elif vrp_val < 0 and cta_val < 0:
                        adjustment *= 1.15

        # 3. Skew 过滤: 极端 skew → 减仓
        if self._skew_panel is not None and product in self._skew_panel.columns:
            if date in self._skew_panel.index:
                skew_val = self._skew_panel.loc[date, product]
                if pd.notna(skew_val) and abs(skew_val) > 0.03:
                    adjustment *= 0.8  # 极端 skew 减仓 20%

        # 4. PCVR: 极端情绪 → 反向
        if self._pcvr_series is not None:
            if date in self._pcvr_series.index:
                pcvr = self._pcvr_series.loc[date]
                if pd.notna(pcvr):
                    if pcvr > self._cfg.pcvr_extreme_threshold:
                        adjustment *= 0.85  # put 情绪极端 → 减仓
                    elif pcvr < 0.7:
                        adjustment *= 0.85  # call 情绪极端 → 减仓

        return np.clip(adjustment, 0.5, 1.5)

    def target_positions(self, date, data) -> dict[str, float]:
        """目标持仓: CTA 信号 × 期权调整系数 → vol_target → 手数."""
        targets = {}

        for product in self._price_panel.columns:
            if date not in self._price_panel.index:
                continue

            # CTA 基础信号
            cta_signal = self._cta_signals.loc[date, product] if date in self._cta_signals.index else 0
            if np.isnan(cta_signal):
                cta_signal = 0

            # 期权因子调整
            adj = self._get_options_adjustment(product, date)
            adjusted_signal = cta_signal * adj

            # vol_target
            vol = self._price_panel[product].pct_change().rolling(self._cfg.vol_lookback).std()
            vol_val = vol.loc[date] if date in vol.index else 0.02
            if np.isnan(vol_val) or vol_val <= 0:
                vol_val = 0.02

            weight = adjusted_signal * (self._cfg.target_vol / vol_val)
            weight = np.clip(weight, -self._cfg.max_leverage, self._cfg.max_leverage)

            # 转为手数
            if abs(weight) > 1e-9:
                spec = self._registry.get_or_default(product)
                price = self._price_panel.loc[date, product]
                if price > 0:
                    notional = price * spec.contract_multiplier
                    capital = self._cfg.initial_cash / max(1, len(self._price_panel.columns))
                    lots = round(weight * capital / notional)
                    if lots != 0:
                        targets[product] = lots

        return targets

    @property
    def _registry(self):
        if not hasattr(self, "_registry_obj"):
            self._registry_obj = CtaContractRegistry()
        return self._registry_obj


def run_options_enhanced_cta(
    products: list[str],
    config: OptionsEnhancedCTAConfig | None = None,
) -> CtaBacktestResult:
    """运行期权增强 CTA 回测."""
    cfg = config or OptionsEnhancedCTAConfig(products=products)
    loader = CtaDataLoader()

    # 加载价格面板
    price_data = {}
    for prod in products:
        try:
            px = loader.load_continuous_main(prod)
            if px is not None and not px.empty:
                px["trade_date"] = pd.to_datetime(px["trade_date"])
                px = px.set_index("trade_date")
                price_data[prod] = px["close"]
        except Exception:
            pass

    if not price_data:
        raise ValueError("无可用价格数据")

    price_panel = pd.DataFrame(price_data)
    price_panel = price_panel.dropna(how="all")

    # 加载期权因子数据
    iv_panel = None
    vrp_panel = None
    try:
        iv_panel = load_iv_panel(start=cfg.start, end=cfg.end)
    except Exception:
        pass

    try:
        vrp_panel = load_vrp_panel(products=products, start=cfg.start, end=cfg.end)
    except Exception:
        pass

    # 创建策略
    strategy = OptionsEnhancedCTA(
        price_panel=price_panel,
        config=cfg,
        iv_panel=iv_panel,
        vrp_panel=vrp_panel,
    )

    # 运行回测
    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    return run_cta_backtest(
        dates=list(price_panel.index),
        prices={p: price_panel[p].tolist() for p in products},
        products={p: p for p in products},
        strategy=strategy,
        config=CtaBacktestConfig(
            initial_cash=cfg.initial_cash,
            commission_rate=0.0001,
            margin_call=False,
        ),
        registry=registry,
        risk=risk,
    )
