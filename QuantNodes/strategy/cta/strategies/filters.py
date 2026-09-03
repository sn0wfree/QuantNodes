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
        """返回 (T,N) 仓位系数矩阵."""
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

            # ER < threshold → 震荡市 → 降仓
            for idx in er.index:
                if pd.notna(er.loc[idx]) and er.loc[idx] < self.er_threshold:
                    coef.loc[idx, product] = self.chop_coef

        return coef
