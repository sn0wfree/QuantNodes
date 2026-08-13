# coding=utf-8
"""日历效应因子.

基于交易日历的商品期货日历效应:
    - SeasonalityFactor: 月份季节性 (同月历史平均收益, expanding 无前视)
    - WeekdayFactor: 周内效应 (周一/周五, expanding 无前视)
    - MonthTurnFactor: 月末/月初效应
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class SeasonalityFactor(Factor):
    """月份季节性因子.

    计算每个自然月的历史平均收益, 当月份历史收益显著为正 → 当月做多.
    研报: 华泰金工《商品期货日历效应》, 农产品/能源季节性.

    参数:
        min_years: 最少年数 (默认 2, 样本不足则无信号)
        threshold: 历史同月收益阈值 (绝对收益)
    """

    name = "seasonality"

    def __init__(self, min_years: int = 2, threshold: float = 0.0):
        self.min_years = min_years
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change()
        month = prices.index.month
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            s = ret[col]
            for m in range(1, 13):
                mask = s.index.month == m
                # expanding 因果统计: 截至当日的历史 m 月平均收益 (无前视)
                exp_sum = s.where(mask).expanding().sum()
                exp_cnt = pd.Series(mask, index=s.index).expanding().sum()
                avg = exp_sum / exp_cnt.replace(0, np.nan)
                cur_month = month == m
                valid = (exp_cnt >= self.min_years * 10) & cur_month
                sig.loc[valid & (avg > self.threshold), col] = 1.0
                sig.loc[valid & (avg < -self.threshold), col] = -1.0
        return sig


class WeekdayFactor(Factor):
    """周内效应因子.

    基于历史各交易日 (周一~周五) 平均收益的方向.
    研报: 华泰金工周内效应.

    参数:
        min_weeks: 最少周数
        threshold: 日均收益阈值
    """

    name = "weekday"

    def __init__(self, min_weeks: int = 50, threshold: float = 0.0):
        self.min_weeks = min_weeks
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change()
        dow = prices.index.dayofweek
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            s = ret[col]
            for d in range(5):
                mask = s.index.dayofweek == d
                # expanding 因果统计: 截至当日的历史 d 日平均收益 (无前视)
                exp_sum = s.where(mask).expanding().sum()
                exp_cnt = pd.Series(mask, index=s.index).expanding().sum()
                avg = exp_sum / exp_cnt.replace(0, np.nan)
                cur_dow = dow == d
                valid = (exp_cnt >= self.min_weeks) & cur_dow
                sig.loc[valid & (avg > self.threshold), col] = 1.0
                sig.loc[valid & (avg < -self.threshold), col] = -1.0
        return sig


class MonthTurnFactor(Factor):
    """月末/月初效应因子.

    月度最后 3 个交易日与最初 3 个交易日的收益差异.
    研报: 调仓周期/交割效应.

    参数:
        window: 月初/月末窗口天数
        threshold: 收益差阈值
    """

    name = "month_turn"

    def __init__(self, window: int = 3, threshold: float = 0.0):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            for _, group in ret[col].groupby(prices.index.to_period("M")):
                if len(group) < self.window * 2:
                    continue
                # 月末 window 天平均收益
                month_end_avg = group.iloc[-self.window:].mean()
                # 月初 window 天平均收益 (需要上月数据, 简化用当月前 window 天)
                month_start_avg = group.iloc[:self.window].mean()
                diff = month_end_avg - month_start_avg
                # 对月末最后 window 天赋值
                last_idx = group.index[-self.window:]
                if diff > self.threshold:
                    sig.loc[last_idx, col] = 1.0
                elif diff < -self.threshold:
                    sig.loc[last_idx, col] = -1.0
        return sig
