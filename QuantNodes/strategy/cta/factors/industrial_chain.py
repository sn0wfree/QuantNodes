# coding=utf-8
"""产业链上下游关系因子.

基于连续主力面板, 利用产业链上下游价格关系:
    - ChainMarginFactor: 盘面利润因子 (下游 - 上游×系数)
    - PriceRatioFactor: 比价因子 (A/B 比值 z-score)
    - MarginMomentumFactor: 利润动量因子 (利润变化方向)
    - LeadLagFactor: 上下游传导因子 (上游对下游滚动相关)

内置主流产业链配方 (见 docs/93-CTA因子调研合集.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor

# 产业链配方: 名称 → (多头品种列表, 空头品种列表, 系数列表)
# 利润 = Σ(mult_i × price_i) - Σ(short_j × price_j)
CHAIN_FORMULAS: dict[str, dict] = {
    "steel_profit_rb": {
        "name": "螺纹盘面利润", "long": ["RB"], "short": ["I", "J"],
        "coef_long": [1.0], "coef_short": [1.6, 0.5],
    },
    "steel_profit_hc": {
        "name": "热卷盘面利润", "long": ["HC"], "short": ["I", "J"],
        "coef_long": [1.0], "coef_short": [1.6, 0.5],
    },
    "coke_profit": {
        "name": "焦化利润", "long": ["J"], "short": ["JM"],
        "coef_long": [1.0], "coef_short": [1.33],
    },
    "pta_fee": {
        "name": "PTA加工费", "long": ["TA"], "short": ["PX"],
        "coef_long": [1.0], "coef_short": [0.655],
    },
    "polyester_profit": {
        "name": "聚酯利润", "long": ["PF"], "short": ["TA", "EG"],
        "coef_long": [1.0], "coef_short": [0.855, 0.13],
    },
    "mto_profit": {
        "name": "MTO利润", "long": ["PP"], "short": ["MA"],
        "coef_long": [1.0], "coef_short": [3.0],
    },
    "crush_profit": {
        "name": "大豆压榨利润", "long": ["M", "Y"], "short": ["A"],
        "coef_long": [0.78, 0.18], "coef_short": [1.0],
    },
    "hog_feed_cost": {
        "name": "生猪饲料成本", "long": ["LH"], "short": ["C", "M"],
        "coef_long": [1.0], "coef_short": [2.45, 1.05],
    },
}


def chain_margin_series(
    prices: pd.DataFrame,
    formula: str,
) -> pd.Series:
    """按配方计算产业链利润序列.

    Args:
        prices: (T, N) 价格面板
        formula: 配方名 (CHAIN_FORMULAS key)

    Returns:
        pd.Series, index 与 prices 对齐, 缺失品种时 NaN
    """
    if formula not in CHAIN_FORMULAS:
        raise KeyError(f"formula '{formula}' not in CHAIN_FORMULAS: {list(CHAIN_FORMULAS)}")
    f = CHAIN_FORMULAS[formula]

    long_val = pd.Series(0.0, index=prices.index)
    for prod, coef in zip(f["long"], f["coef_long"]):
        if prod not in prices.columns:
            return pd.Series(np.nan, index=prices.index)
        long_val = long_val + coef * prices[prod]

    short_val = pd.Series(0.0, index=prices.index)
    for prod, coef in zip(f["short"], f["coef_short"]):
        if prod not in prices.columns:
            return pd.Series(np.nan, index=prices.index)
        short_val = short_val + coef * prices[prod]

    return long_val - short_val


def _zscore_series(series: pd.Series, window: int) -> pd.Series:
    """滚动 z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


class ChainMarginFactor(Factor):
    """盘面利润因子.

    产业链利润 = Σ(多头×系数) - Σ(空头×系数).
    利润 z-score 高 → 利润回归 → 空头; 低 → 多头.
    研报: 黑色/油粕/化工产业链利润因子.

    参数:
        formula: 配方名
        window: z-score 窗口
        z_threshold: 触发阈值
    """

    name = "chain_margin"

    def __init__(self, formula: str, window: int = 60, z_threshold: float = 1.0):
        self.formula = formula
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        margin = chain_margin_series(prices, self.formula)
        z = _zscore_series(margin, self.window)
        sig = pd.DataFrame(0.0, index=prices.index, columns=[self.formula])
        sig.loc[z > self.z_threshold, self.formula] = -1.0  # 高利润 → 空
        sig.loc[z < -self.z_threshold, self.formula] = 1.0  # 低利润 → 多
        return sig


class PriceRatioFactor(Factor):
    """比价因子.

    两个品种比值 A/B 的 z-score.
    比价偏离均值 → 回归; 金银比/铜铝比等.
    研报: 有色/贵金属比价策略.

    参数:
        num_product: 分子品种
        den_product: 分母品种
        window: z-score 窗口
        z_threshold: 触发阈值
        inverse: 比值高 → 多 (默认 False: 高 → 空)
    """

    name = "price_ratio"

    def __init__(self, num_product: str, den_product: str, window: int = 60,
                 z_threshold: float = 1.0, inverse: bool = False):
        self.num_product = num_product
        self.den_product = den_product
        self.window = window
        self.z_threshold = z_threshold
        self.inverse = inverse

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.num_product not in prices.columns or self.den_product not in prices.columns:
            raise KeyError(
                f"products {self.num_product}/{self.den_product} not in panel: "
                f"{list(prices.columns)}"
            )
        ratio = prices[self.num_product] / prices[self.den_product].replace(0, np.nan)
        z = _zscore_series(ratio, self.window)
        col = f"{self.num_product}_{self.den_product}_ratio"
        sig = pd.DataFrame(0.0, index=prices.index, columns=[col])
        if self.inverse:
            sig.loc[z > self.z_threshold, col] = 1.0
            sig.loc[z < -self.z_threshold, col] = -1.0
        else:
            sig.loc[z > self.z_threshold, col] = -1.0
            sig.loc[z < -self.z_threshold, col] = 1.0
        return sig


class MarginMomentumFactor(Factor):
    """利润动量因子.

    利润变化方向: 利润上升 → 多头, 下降 → 空头.
    研报: 利润趋势持续性 (与利润均值回归互补).

    参数:
        formula: 配方名
        window: 利润变化窗口
    """

    name = "margin_momentum"

    def __init__(self, formula: str, window: int = 10):
        self.formula = formula
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        margin = chain_margin_series(prices, self.formula)
        change = margin.diff(self.window)
        sig = pd.DataFrame(0.0, index=prices.index, columns=[self.formula])
        sig.loc[change > 0, self.formula] = 1.0
        sig.loc[change < 0, self.formula] = -1.0
        return sig


class LeadLagFactor(Factor):
    """上下游传导因子.

    上游收益对下游收益的滚动相关 (含滞后):
    正相关 → 传导顺畅 → 上下游同向.
    研报: 产业链成本传导时滞研究.

    参数:
        upstream: 上游品种
        downstream: 下游品种
        window: 相关窗口
        lag: 上游领先天数
        threshold: 相关阈值
    """

    name = "lead_lag"

    def __init__(self, upstream: str, downstream: str, window: int = 60,
                 lag: int = 1, threshold: float = 0.3):
        self.upstream = upstream
        self.downstream = downstream
        self.window = window
        self.lag = lag
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.upstream not in prices.columns or self.downstream not in prices.columns:
            raise KeyError(
                f"products {self.upstream}/{self.downstream} not in panel: "
                f"{list(prices.columns)}"
            )
        up_ret = prices[self.upstream].pct_change()
        dn_ret = prices[self.downstream].pct_change()
        # 上游 t-lag 日收益 vs 下游 t 日收益
        corr = up_ret.shift(self.lag).rolling(self.window).corr(dn_ret)
        col = f"{self.upstream}->{self.downstream}"
        sig = pd.DataFrame(0.0, index=prices.index, columns=[col])
        sig.loc[corr > self.threshold, col] = 1.0
        sig.loc[corr < -self.threshold, col] = -1.0
        return sig
