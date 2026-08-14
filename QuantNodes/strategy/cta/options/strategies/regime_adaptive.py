# coding=utf-8
"""Regime-Adaptive 期权策略: 根据市场状态路由不同策略.

基于 期权工作平台 v3 Regime switching 研究:
    低波+不跌(0) → 卖 Put Spread (收 theta)
    低波+跌(1)   → 买 Put Spread (保护)
    高波+不跌(2) → 卖 Strangle (高 IV 收取)
    高波+跌(3)   → 买 Long Put (对冲)

使用方法:
    from ..strategies import get_strategy
    s = get_strategy("regime_adaptive")
    targets = s.select("RB", date, spot, None, loader=..., vol_signal_gen=...)
"""
from __future__ import annotations

import pandas as pd

from . import OptionStrategy, StrategyConfig, register
from ..selector import select_call, select_put, select_put_spread
from ..vol_signals import Regime, VolSignalGenerator


@register
class RegimeAdaptive(OptionStrategy):
    """Regime 自适应策略: 根据 IV 分位 + HV/IV 路由不同期权策略."""

    name = "regime_adaptive"
    description = (
        "低波不跌→卖put spread, 低波跌→买put spread, "
        "高波不跌→卖strangle, 高波跌→买long put"
    )

    def __init__(
        self,
        config: StrategyConfig | None = None,
        # put spread 参数
        buy_put_delta: float = 0.25,
        sell_put_delta: float = 0.10,
        # strangle 参数
        strangle_call_delta: float = 0.25,
        strangle_put_delta: float = 0.25,
        # regime 阈值
        iv_pct_high: float = 0.5,
        hv_iv_drop_threshold: float = 1.0,
    ):
        super().__init__(config)
        self._buy_put_delta = buy_put_delta
        self._sell_put_delta = sell_put_delta
        self._strangle_call_delta = strangle_call_delta
        self._strangle_put_delta = strangle_put_delta
        self._iv_pct_high = iv_pct_high
        self._hv_iv_drop_threshold = hv_iv_drop_threshold

    def select(self, product, date, underlying_price, existing_holding, **kwargs):
        """根据 regime 选择对应的期权策略."""
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")
        vol_signal_gen: VolSignalGenerator | None = kwargs.get("vol_signal_gen")

        if existing_holding is not None:
            return {}

        # 获取 regime
        regime = self._get_regime(product, date, vol_signal_gen)

        # 无信号时不做任何操作
        if vol_signal_gen is None:
            return {}

        # 按 regime 路由
        if regime == Regime.LOW_VOL_NO_DROP:
            return self._sell_put_spread(product, date, underlying_price, loader, um, oe)
        elif regime == Regime.LOW_VOL_DROP:
            return self._buy_put_spread(product, date, underlying_price, loader, um, oe)
        elif regime == Regime.HIGH_VOL_NO_DROP:
            return self._sell_strangle(product, date, underlying_price, loader, um, oe)
        elif regime == Regime.HIGH_VOL_DROP:
            return self._buy_long_put(product, date, underlying_price, loader, um, oe)
        return {}

    def _get_regime(
        self,
        product: str,
        date: pd.Timestamp,
        vol_signal_gen: VolSignalGenerator | None,
    ) -> Regime:
        """获取当前 regime."""
        if vol_signal_gen is None:
            return Regime.LOW_VOL_NO_DROP  # 默认不操作
        return vol_signal_gen.compute_regime(product, date)

    def _sell_put_spread(self, product, date, spot, loader, um, oe):
        """低波不跌: 卖 put spread (收 theta)."""
        long_put = select_put(
            product, date, spot, target_delta=self._buy_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        short_put = select_put(
            product, date, spot, target_delta=self._sell_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if long_put is not None:
            targets[long_put.symbol] = 1.0   # 买
        if short_put is not None and short_put.strike < (long_put.strike if long_put else 0):
            targets[short_put.symbol] = -1.0  # 卖
        return targets

    def _buy_put_spread(self, product, date, spot, loader, um, oe):
        """低波跌: 买 put spread (保护)."""
        long_put = select_put(
            product, date, spot, target_delta=self._buy_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        short_put = select_put(
            product, date, spot, target_delta=self._sell_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if long_put is not None:
            targets[long_put.symbol] = 1.0
        if short_put is not None and short_put.strike < (long_put.strike if long_put else 0):
            targets[short_put.symbol] = -1.0
        return targets

    def _sell_strangle(self, product, date, spot, loader, um, oe):
        """高波不跌: 卖 strangle (高 IV 收取)."""
        call = select_call(
            product, date, spot, target_delta=self._strangle_call_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        put = select_put(
            product, date, spot, target_delta=self._strangle_put_delta,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if call is not None:
            targets[call.symbol] = -1.0
        if put is not None:
            targets[put.symbol] = -1.0
        return targets

    def _buy_long_put(self, product, date, spot, loader, um, oe):
        """高波跌: 买 long put (对冲)."""
        put = select_put(
            product, date, spot, target_delta=0.25,
            min_days=self._cfg.min_days, max_days=self._cfg.max_days,
            loader=loader, um=um, oe=oe,
        )
        if put is not None:
            return {put.symbol: 1.0}
        return {}
