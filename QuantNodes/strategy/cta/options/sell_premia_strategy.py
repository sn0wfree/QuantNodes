# coding=utf-8
"""Sell Premia with CTA: 卖方为主 + CTA 方向保护.

核心逻辑:
    基底: 卖 put spread / iron condor (收 theta)
    CTA 保护: 强趋势时减仓 (减少方向风险)
    VRP 加仓: 高 VRP 时加仓 (更多 theta 可收)
    止损: 单笔亏损超阈值时平仓

收益来源: Theta 衰减 (期权时间价值)
风险控制: CTA 方向过滤 + VRP 仓位调整 + 止损
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..options.selector import select_call, select_put
from ..options.vol_signals import VolSignalGenerator


@dataclass
class SellPremiaConfig:
    """卖方策略配置."""
    products: list[str] = field(default_factory=lambda: ["RB"])
    initial_cash: float = 1_000_000.0
    max_margin_pct: float = 0.50

    # 卖方参数
    put_delta: float = 0.15        # 卖 put delta (OTM)
    call_delta: float = 0.15       # 卖 call delta (OTM)
    wing_delta: float = 0.05       # 保护腿 delta

    # CTA 保护
    cta_reduce_threshold: float = 0.5  # |CTA| > 此值时减仓
    cta_reduce_pct: float = 0.5        # 减仓比例

    # VRP 加仓
    vrp_high_threshold: float = 0.6    # VRP 分位 > 此值时加仓
    vrp_boost_pct: float = 0.2         # 加仓比例

    # 换月/持仓
    roll_days: int = 10
    min_dte: int = 30
    max_dte: int = 60
    min_hold_days: int = 15

    # 止损
    stop_loss_pct: float = 0.02   # 单笔亏损 > 2% NAV 时止损


class SellPremiaWithCTA:
    """卖方为主 + CTA 方向保护策略.

    基底: 卖 put spread (看涨/中性) 或 iron condor (中性)
    CTA: 强趋势时减仓
    VRP: 高 VRP 时加仓
    """

    def __init__(self, config: SellPremiaConfig | None = None):
        self._cfg = config or SellPremiaConfig()
        self._loader = None
        self._um = None
        self._oe = None

    def select(
        self,
        product: str,
        date: pd.Timestamp,
        spot: float,
        cta_signal: float | None = None,
        vrp_pct: float | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """选择卖方策略.

        Args:
            cta_signal: CTA 趋势信号 (-1 ~ +1)
            vrp_pct: VRP 分位 (0~1)

        Returns:
            {option_symbol: qty} (负=卖)
        """
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        cta_strength = abs(cta_signal) if cta_signal is not None else 0

        # CTA 强趋势时减仓 (不卖方向性太强的期权)
        if cta_strength > self._cfg.cta_reduce_threshold:
            # 强趋势: 只卖反方向 spread (对冲风险)
            if cta_signal > 0:
                # 强看涨 → 卖 call spread (对冲上行风险)
                return self._sell_call_spread(product, date, spot, loader, um, oe)
            else:
                # 强看跌 → 卖 put spread (对冲下行风险)
                return self._sell_put_spread(product, date, spot, loader, um, oe)
        else:
            # 弱趋势/中性: 卖 iron condor (双边收 theta)
            return self._sell_iron_condor(product, date, spot, loader, um, oe)

    def get_position_size(
        self,
        base_notional: float,
        vrp_pct: float | None = None,
        cta_strength: float = 0.0,
    ) -> float:
        """计算仓位大小.

        base_notional × vrp_mult × cta_mult
        """
        # VRP 加仓: 高 VRP → 更多 theta 可收
        vrp_mult = 1.0
        if vrp_pct is not None and vrp_pct > self._cfg.vrp_high_threshold:
            vrp_mult = 1.0 + self._cfg.vrp_boost_pct

        # CTA 减仓: 强趋势 → 减少卖方仓位
        cta_mult = 1.0 - min(1.0, cta_strength) * self._cfg.cta_reduce_pct

        return base_notional * vrp_mult * cta_mult

    def _sell_put_spread(self, product, date, spot, loader, um, oe):
        """卖 put spread: 卖高 delta put + 买低 delta put."""
        sell_put = select_put(
            product, date, spot, target_delta=self._cfg.put_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        buy_put = select_put(
            product, date, spot, target_delta=self._cfg.wing_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if sell_put:
            targets[sell_put.symbol] = -1.0  # 卖
        if buy_put and buy_put.strike < (sell_put.strike if sell_put else 0):
            targets[buy_put.symbol] = 1.0    # 买保护
        return targets

    def _sell_call_spread(self, product, date, spot, loader, um, oe):
        """卖 call spread: 卖低 delta call + 买高 delta call."""
        sell_call = select_call(
            product, date, spot, target_delta=self._cfg.call_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        buy_call = select_call(
            product, date, spot, target_delta=self._cfg.wing_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if sell_call:
            targets[sell_call.symbol] = -1.0  # 卖
        if buy_call and buy_call.strike > (sell_call.strike if sell_call else 0):
            targets[buy_call.symbol] = 1.0    # 买保护
        return targets

    def _sell_iron_condor(self, product, date, spot, loader, um, oe):
        """卖 iron condor: 卖 put spread + 卖 call spread."""
        targets = {}

        # Put spread leg
        sell_put = select_put(
            product, date, spot, target_delta=self._cfg.put_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        buy_put = select_put(
            product, date, spot, target_delta=self._cfg.wing_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        if sell_put:
            targets[sell_put.symbol] = -1.0
        if buy_put and buy_put.strike < (sell_put.strike if sell_put else 0):
            targets[buy_put.symbol] = 1.0

        # Call spread leg
        sell_call = select_call(
            product, date, spot, target_delta=self._cfg.call_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        buy_call = select_call(
            product, date, spot, target_delta=self._cfg.wing_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        if sell_call:
            targets[sell_call.symbol] = -1.0
        if buy_call and buy_call.strike > (sell_call.strike if sell_call else 0):
            targets[buy_call.symbol] = 1.0

        return targets
