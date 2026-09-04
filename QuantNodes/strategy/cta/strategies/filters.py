# coding=utf-8
"""波动率/市场状态过滤层 — 输出全局仓位系数 (T,N) ∈ [0,1].

从 factor_modules.py 拆出 (pure move, 逻辑不变):
    - VolRiskFilter:      市场级危机过滤 (全市场 HV z>阈值 → 全局降仓)
    - MarketRegimeFilter: 震荡市识别 (效率比 ER) → 降仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class VolRiskFilter:
    """波动率风险过滤层: 输出全局仓位系数 (T,) ∈ [0,1].

    市场级危机过滤 (业界标准, 实验增益): 全市场平均 HV 滚动 z-score
    超过阈值 → 全局降仓. 10 品种实验: z>1.0 → 0.5x 使 Sharpe 1.109→1.190.
    单品种级过滤 (HV 逐品种 z-score) 实验无增益, 默认关闭.
    """

    def __init__(
        self,
        z_window: int = 252,
        mkt_z_threshold: float = 1.0,
        mkt_coef: float = 0.5,
        product_z_threshold: float = 2.0,
        product_coef: float = 0.5,
        apply_per_product: bool = False,
    ):
        self.z_window = z_window
        self.mkt_z_threshold = mkt_z_threshold
        self.mkt_coef = mkt_coef
        self.product_z_threshold = product_z_threshold
        self.product_coef = product_coef
        self.apply_per_product = apply_per_product

    def coefficients(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """返回 (T,N) 仓位系数矩阵."""
        from ..factors.volatility import HVFactor
        hv = HVFactor().compute(price_panel)
        mu = hv.rolling(self.z_window, min_periods=20).mean()
        sd = hv.rolling(self.z_window, min_periods=20).std()
        hv_z = (hv - mu) / sd.replace(0, np.nan)
        coef = pd.DataFrame(1.0, index=price_panel.index, columns=price_panel.columns)
        mkt_z = hv_z.mean(axis=1)
        coef.loc[mkt_z > self.mkt_z_threshold, :] = self.mkt_coef
        if self.apply_per_product:
            coef[hv_z > self.product_z_threshold] = self.product_coef
        return coef


class MarketRegimeFilter:
    """市场状态过滤层: 识别震荡市并降低仓位.

    检测方法:
    1. 价格效率比 (Efficiency Ratio): 价格移动距离 / 总波动
       - ER高 → 趋势市 (价格单向移动)
       - ER低 → 震荡市 (价格来回波动)
    2. 价格位置: 价格在布林带中轨附近 → 震荡市

    输出: 仓位系数 (T,N) ∈ [0,1]
    - 趋势市: 1.0 (满仓)
    - 震荡市: 0.3-0.5 (降仓)
    """

    def __init__(
        self,
        er_window: int = 10,
        er_threshold: float = 0.4,
        chop_coef: float = 0.4,
    ):
        self.er_window = er_window
        self.er_threshold = er_threshold
        self.chop_coef = chop_coef

    def coefficients(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """返回 (T,N) 仓位系数矩阵. (向量化: 逻辑与逐日循环版逐位等价)"""
        coef = pd.DataFrame(1.0, index=price_panel.index, columns=price_panel.columns)

        for product in price_panel.columns:
            close = price_panel[product].dropna()
            if len(close) < self.er_window * 2:
                continue

            # 计算效率比 (Efficiency Ratio)
            # ER = |价格变化| / 总波动
            price_change = close.diff(self.er_window).abs()
            total_vol = close.diff().abs().rolling(self.er_window).sum()
            er = price_change / total_vol.replace(0, np.nan)

            # ER < threshold → 震荡市 → 降仓 (NaN → False, 与 pd.notna 守卫等价)
            chop = er < self.er_threshold
            coef.loc[chop[chop].index, product] = self.chop_coef

        return coef


class ADXRegimeFilter:
    """ADX 滞回过滤层: ADX>enter → 满仓; ADX<exit → 降仓; 中间维持前状态.

    滞回 (enter≠exit) 防止 ADX 在阈值附近抖动导致过滤系数频繁开关 (churn).
    无 OHLC 数据的品种系数保持 1.0 (不降仓, 与 ER 版 NaN 语义一致).
    """

    def __init__(
        self,
        window: int = 14,
        enter_threshold: float = 25.0,
        exit_threshold: float = 20.0,
        chop_coef: float = 0.4,
    ):
        self.window = window
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.chop_coef = chop_coef

    def coefficients(
        self, price_panel: pd.DataFrame, ohlc_map: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """返回 (T,N) 仓位系数矩阵. 需逐品种 OHLC (ADX 依赖 high/low)."""
        coef = pd.DataFrame(1.0, index=price_panel.index, columns=price_panel.columns)
        if not ohlc_map:
            return coef

        from ..factors.indicators import adx as adx_fn

        for product in price_panel.columns:
            ohlc = ohlc_map.get(product)
            if ohlc is None:
                continue
            adx_vals = adx_fn(ohlc, self.window)
            adx_vals = adx_vals.reindex(price_panel.index)

            # 滞回状态机 (ffill 向量化): >enter→满仓, <exit→降仓, 中间维持
            raw = pd.Series(np.nan, index=adx_vals.index)
            raw[adx_vals > self.enter_threshold] = 1.0
            raw[adx_vals < self.exit_threshold] = self.chop_coef
            state = raw.ffill().fillna(1.0)
            coef[product] = state

        return coef
