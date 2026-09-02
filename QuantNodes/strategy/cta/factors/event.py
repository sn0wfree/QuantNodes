# coding=utf-8
"""事件/跳变因子.

基于价格极端变动的统计:
    - ExtremeMoveFactor: 极端涨跌事件 (单日大幅波动)
    - GapStreakFactor: 连续跳空统计
    - LimitMoveFactor: 近似涨跌停事件 (接近涨跌停幅度)
"""
from __future__ import annotations

import pandas as pd

from .base import Factor


class ExtremeMoveFactor(Factor):
    """极端波动因子.

    单日 |收益| 超过 N 日均值的 M 倍 → 极端事件.
    极端上涨后 → 动量延续/反转 (参数化); 研报: 事件驱动.

    参数:
        window: 均收益窗口
        mult: 极端倍数
        direction: 'both'/'up'/'down'
    """

    name = "extreme_move"

    def __init__(self, window: int = 20, mult: float = 3.0,
                 direction: str = "both"):
        self.window = window
        self.mult = mult
        self.direction = direction

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        avg = ret.abs().rolling(self.window).mean()
        extreme_up = (ret > avg * self.mult) & (ret > 0)
        extreme_down = (ret < -avg * self.mult) & (ret < 0)

        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if self.direction in ("both", "up"):
            sig[extreme_up] = 1.0    # 极端上涨 → 多头 (动量延续)
        if self.direction in ("both", "down"):
            sig[extreme_down] = -1.0  # 极端下跌 → 空头
        return sig


class GapStreakFactor(Factor):
    """连续跳空因子.

    统计连续同向跳空天数: 连续 N 日同向跳空 → 强趋势/衰竭信号.
    研报: 跳空缺口策略.

    参数:
        direction: 'up'/'down'/'both'
        streak_threshold: 触发连续天数
    """

    name = "gap_streak"

    def __init__(self, direction: str = "both", streak_threshold: int = 3):
        self.direction = direction
        self.streak_threshold = streak_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        gap = prices.diff() / prices.shift(1)
        up_gap = (gap > 0).astype(int)
        down_gap = (gap < 0).astype(int)

        def _streak(s: pd.Series) -> pd.Series:
            """连续 1 的计数."""
            g = (s != s.shift()).cumsum()
            return s.groupby(g).cumsum()

        up_streak = up_gap.apply(_streak)
        down_streak = down_gap.apply(_streak)

        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if self.direction in ("both", "up"):
            sig[up_streak >= self.streak_threshold] = 1.0
        if self.direction in ("both", "down"):
            sig[down_streak >= self.streak_threshold] = -1.0
        return sig


class LimitMoveFactor(Factor):
    """近似涨跌停因子.

    单日收益接近涨跌停幅度 (如 ±5%) → 封板事件.
    商品期货涨跌停约 4%-10% (品种不同).
    研报: 涨跌停板事件效应.

    参数:
        limit_pct: 近似涨跌停幅度 (0.05 = 5%)
        tolerance: 允许偏差
    """

    name = "limit_move"

    def __init__(self, limit_pct: float = 0.05, tolerance: float = 0.005):
        self.limit_pct = limit_pct
        self.tolerance = tolerance

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        near_limit_up = ret > (self.limit_pct - self.tolerance)
        near_limit_down = ret < -(self.limit_pct - self.tolerance)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[near_limit_up] = 1.0
        sig[near_limit_down] = -1.0
        return sig
