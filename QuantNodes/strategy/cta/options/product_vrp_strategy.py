# coding=utf-8
"""Product VRP Strategy: 产品分层 VRP 策略.

根据产品 VRP 特性, 不同产品用不同期权策略:
    - VRP 正 (IV > HV): 卖 put spread (IV 高估, 收 theta)
    - VRP 负 (IV < HV): 买 put spread (IV 低估, 买便宜)
    - VRP 中性: Calendar Spread (期限结构)

VRP 分类来自 docs/100:
    正 VRP: AU, L, PP, RU, CU (mean_VRP > 0.01)
    负 VRP: I, M, FG, SA (mean_VRP < -0.01)
    中性: SR, MA, TA
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..options.selector import select_call, select_put
from ..options.hedge_data import load_vrp_panel


# VRP 分类映射 (静态, 基于 docs/100 研究)
VRP_CLASSIFICATION = {
    "positive": ["AU", "L", "PP", "RU", "CU", "FU", "V", "AG"],
    "negative": ["I", "M", "FG", "SA", "JD", "B"],
    "neutral": ["SR", "MA", "TA", "RB", "ZC", "SF", "SM"],
}


@dataclass
class ProductVRPConfig:
    """产品 VRP 策略配置."""
    products: list[str] = field(default_factory=lambda: ["RB"])
    initial_cash: float = 1_000_000.0
    max_margin_pct: float = 0.50

    # 策略参数
    sell_delta: float = 0.15       # 卖方 delta
    buy_delta: float = 0.25        # 买方 delta
    wing_delta: float = 0.05       # 保护腿 delta
    calendar_near_delta: float = 0.25
    calendar_far_delta: float = 0.25

    # 换月/持仓
    roll_days: int = 10
    min_dte: int = 30
    max_dte: int = 60
    near_min_dte: int = 15
    near_max_dte: int = 35
    far_min_dte: int = 45
    far_max_dte: int = 75
    min_hold_days: int = 15

    # 动态分类
    use_dynamic_vrp: bool = True   # 是否用滚动 VRP 动态分类
    vrp_lookback: int = 252        # VRP 分类滚动窗口


class ProductVRPStrategy:
    """产品分层 VRP 策略.

    正 VRP 品种: 卖 put spread (IV > HV, 收 theta)
    负 VRP 品种: 买 put spread (IV < HV, 买便宜)
    中性品种: Calendar Spread (期限结构)
    """

    def __init__(self, config: ProductVRPConfig | None = None):
        self._cfg = config or ProductVRPConfig()
        self._loader = None
        self._um = None
        self._oe = None
        self._vrp_class: dict[str, str] = {}
        self._vrp_panel: pd.DataFrame | None = None

    def load_vrp_classification(self, start: str, end: str):
        """加载 VRP 分类."""
        # 静态分类
        for cls, products in VRP_CLASSIFICATION.items():
            for p in products:
                self._vrp_class[p] = cls

        # 动态分类 (如有 VRP 数据)
        if self._cfg.use_dynamic_vrp:
            try:
                self._vrp_panel = load_vrp_panel(
                    products=self._cfg.products, start=start, end=end,
                )
                self._update_dynamic_classification()
            except Exception:
                pass

    def _update_dynamic_classification(self):
        """基于滚动 VRP 均值动态更新分类."""
        if self._vrp_panel is None:
            return

        for product in self._cfg.products:
            if product not in self._vrp_panel.columns:
                continue
            vrp_series = self._vrp_panel[product].dropna()
            if len(vrp_series) < 60:
                continue

            # 滚动 252 天均值
            mean_vrp = vrp_series.rolling(self._cfg.vrp_lookback, min_periods=60).mean()
            if mean_vrp.empty:
                continue

            last_mean = mean_vrp.iloc[-1]
            if pd.isna(last_mean):
                continue

            if last_mean > 0.01:
                self._vrp_class[product] = "positive"
            elif last_mean < -0.01:
                self._vrp_class[product] = "negative"
            else:
                self._vrp_class[product] = "neutral"

    def get_vrp_class(self, product: str) -> str:
        """获取产品 VRP 分类."""
        return self._vrp_class.get(product, "neutral")

    def select(
        self,
        product: str,
        date: pd.Timestamp,
        spot: float,
        **kwargs,
    ) -> dict[str, float]:
        """根据 VRP 分类选择策略.

        Returns:
            {option_symbol: qty}
        """
        loader = kwargs.get("loader")
        um = kwargs.get("um")
        oe = kwargs.get("oe")

        vrp_cls = self.get_vrp_class(product)

        if vrp_cls == "positive":
            # 正 VRP: IV > HV → 卖 put spread (收 theta)
            return self._sell_put_spread(product, date, spot, loader, um, oe)
        elif vrp_cls == "negative":
            # 负 VRP: IV < HV → 买 put spread (买便宜)
            return self._buy_put_spread(product, date, spot, loader, um, oe)
        else:
            # 中性: Calendar Spread (期限结构)
            return self._calendar_spread(product, date, spot, loader, um, oe)

    def _sell_put_spread(self, product, date, spot, loader, um, oe):
        """卖 put spread (正 VRP)."""
        sell_put = select_put(
            product, date, spot, target_delta=self._cfg.sell_delta,
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
            targets[sell_put.symbol] = -1.0
        if buy_put and buy_put.strike < (sell_put.strike if sell_put else 0):
            targets[buy_put.symbol] = 1.0
        return targets

    def _buy_put_spread(self, product, date, spot, loader, um, oe):
        """买 put spread (负 VRP)."""
        buy_put = select_put(
            product, date, spot, target_delta=self._cfg.buy_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        sell_put = select_put(
            product, date, spot, target_delta=self._cfg.wing_delta,
            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if buy_put:
            targets[buy_put.symbol] = 1.0
        if sell_put and sell_put.strike < (buy_put.strike if buy_put else 0):
            targets[sell_put.symbol] = -1.0
        return targets

    def _calendar_spread(self, product, date, spot, loader, um, oe):
        """Calendar Spread (中性 VRP)."""
        near = select_put(
            product, date, spot, target_delta=self._cfg.calendar_near_delta,
            min_days=self._cfg.near_min_dte, max_days=self._cfg.near_max_dte,
            loader=loader, um=um, oe=oe,
        )
        far = select_put(
            product, date, spot, target_delta=self._cfg.calendar_far_delta,
            min_days=self._cfg.far_min_dte, max_days=self._cfg.far_max_dte,
            loader=loader, um=um, oe=oe,
        )
        targets = {}
        if near:
            targets[near.symbol] = -1.0  # 卖近月
        if far:
            targets[far.symbol] = 1.0    # 买远月
        return targets
