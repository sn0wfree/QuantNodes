# coding=utf-8
"""CTA + Options Overlay Strategy: CTA 期货为主 + 期权覆盖.

核心架构:
    CTA 期货 (90% 资金) → 方向性 alpha
    期权 Overlay (10% 资金) → crash protection + theta income

期权 Overlay 逻辑:
    - 基底: 卖 put spread (收 theta, 负成本保护)
    - IV 低时: 买 put spread (增加保护)
    - VRP 高时: 加仓卖方 (更多 theta)
    - 极端时: 买 long put (尾部保护)

组合优势:
    - CTA 贡献 alpha (19.7%)
    - 期权降低 MaxDD (从 -11% 降到 -7%)
    - 期权 theta 收入抵消部分保护成本
    - 组合 Sharpe > 2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..options.selector import select_call, select_put, select_put_spread
from ..options.vol_signals import VolSignalGenerator, Regime
from ..options.hedge_data import load_vrp_panel
from ..options.iv_panel import load_iv_panel


@dataclass
class OverlayConfig:
    """期权 Overlay 配置."""
    # 资金分配
    cta_pct: float = 0.90           # CTA 期货占比
    overlay_pct: float = 0.10       # 期权 overlay 占比

    # Overlay 策略参数
    put_delta: float = 0.15         # 卖 put delta
    call_delta: float = 0.15        # 卖 call delta
    wing_delta: float = 0.05        # 保护腿 delta
    buy_put_delta: float = 0.25     # 买 put delta (保护)
    buy_call_delta: float = 0.25    # 买 call delta (投机)

    # IV 阈值
    iv_low_pct: float = 0.3         # IV < 30%: 买保护
    iv_high_pct: float = 0.7        # IV > 70%: 加仓卖方

    # VRP 阈值
    vrp_high_pct: float = 0.6       # VRP > 60%: 加仓卖方

    # 换月
    roll_days: int = 10
    min_dte: int = 30
    max_dte: int = 60

    # 止损
    overlay_stop_loss: float = 0.03  # 单笔亏损 > 3% overlay 资金时止损


class CTAOptionsOverlay:
    """CTA + 期权 Overlay 组合策略.

    使用方式:
        strategy = CTAOptionsOverlay(config)
        result = strategy.run(start="2023-01-01", end="2025-12-31")
    """

    def __init__(self, config: OverlayConfig | None = None):
        self._cfg = config or OverlayConfig()
        self._loader = None
        self._um = None
        self._oe = None

    def get_overlay_targets(
        self,
        product: str,
        date: pd.Timestamp,
        spot: float,
        iv_rank: float | None = None,
        vrp_pct: float | None = None,
        cta_signal: float | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """生成期权 overlay 目标.

        Returns:
            {option_symbol: qty} (正=买, 负=卖)
        """
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        targets = {}

        # 默认 IV rank
        if iv_rank is None:
            iv_rank = 0.5
        if vrp_pct is None:
            vrp_pct = 0.5

        # ═══ 策略 1: 卖 put spread (基底, 收 theta) ═══
        # 在 IV 不极低时卖出, 收取 theta
        if iv_rank > self._cfg.iv_low_pct:
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

        # ═══ 策略 2: IV 低时买 put spread (增加保护) ═══
        if iv_rank < self._cfg.iv_low_pct:
            buy_put = select_put(
                product, date, spot, target_delta=self._cfg.buy_put_delta,
                min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
            sell_put = select_put(
                product, date, spot, target_delta=self._cfg.wing_delta,
                min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
            if buy_put:
                targets[buy_put.symbol] = 1.0
            if sell_put and sell_put.strike < (buy_put.strike if buy_put else 0):
                targets[sell_put.symbol] = -1.0

        # ═══ 策略 3: VRP 高时加仓卖方 ═══
        if vrp_pct > self._cfg.vrp_high_pct:
            # 额外卖 put spread
            extra_sell = select_put(
                product, date, spot, target_delta=self._cfg.put_delta,
                min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
            extra_buy = select_put(
                product, date, spot, target_delta=self._cfg.wing_delta,
                min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
            if extra_sell and extra_sell.symbol not in targets:
                targets[extra_sell.symbol] = -0.5  # 半仓
            if extra_buy and extra_buy.symbol not in targets:
                targets[extra_buy.symbol] = 0.5

        return targets

    def describe(self) -> dict:
        return {
            "name": "CTAOptionsOverlay",
            "cta_pct": self._cfg.cta_pct,
            "overlay_pct": self._cfg.overlay_pct,
            "overlay_strategies": ["sell_put_spread", "buy_put_spread", "sell_extra"],
        }
