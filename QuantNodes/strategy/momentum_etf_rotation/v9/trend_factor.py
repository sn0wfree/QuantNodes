# coding=utf-8
"""v9 趋势因子 — 三层信号融合 (短期动量 + 中期趋势 + 长期趋势).

T(t) ∈ [0.7, 1.3]:
    趋势上行 → T > 1 (超配)
    趋势下行 → T < 1 (低配)
    趋势强度决定调整幅度
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_factor(returns_series, short=13, medium=26, long=52):
    """计算单个资产的趋势因子.

    参数:
        returns_series: 日收益率 Series
        short: 短期窗口 (周)
        medium: 中期窗口 (周)
        long: 长期窗口 (周)

    返回:
        trend_factor: 趋势调整系数 ∈ [0.7, 1.3]
        trend_score: 趋势评分 ∈ [-1, 1]
        components: 各层信号
    """
    # 转为日频
    short_d = short * 5
    medium_d = medium * 5
    long_d = long * 5

    if len(returns_series.dropna()) < long_d:
        return 1.0, 0.0, {}

    # 1. 短期动量 (13周 ≈ 3月)
    mom_short = returns_series.rolling(short_d).mean()
    sig_short = np.sign(mom_short.iloc[-1]) if not np.isnan(mom_short.iloc[-1]) else 0

    # 2. 中期趋势 (26周 ≈ 6月) — MA 交叉
    nav = (1 + returns_series).cumprod()
    ma_fast = nav.rolling(medium_d // 2).mean()
    ma_slow = nav.rolling(medium_d).mean()
    sig_medium = np.sign(ma_fast.iloc[-1] - ma_slow.iloc[-1]) if not np.isnan(ma_fast.iloc[-1]) else 0

    # 3. 长期趋势 (52周 ≈ 1年) — 价格位置
    ma_long = nav.rolling(long_d).mean()
    sig_long = np.sign(nav.iloc[-1] - ma_long.iloc[-1]) if not np.isnan(ma_long.iloc[-1]) else 0

    # 加权融合
    trend_score = 0.4 * sig_short + 0.35 * sig_medium + 0.25 * sig_long

    # 趋势强度
    trend_strength = abs(trend_score)

    # 转换为调整系数
    trend_factor = 1.0 + trend_score * 0.3
    trend_factor = np.clip(trend_factor, 0.7, 1.3)

    return float(trend_factor), float(trend_score), {
        'short': float(sig_short),
        'medium': float(sig_medium),
        'long': float(sig_long),
        'strength': float(trend_strength),
    }


def compute_all_trend_factors(returns_df, short=13, medium=26, long=52):
    """对所有资产计算趋势因子.

    参数:
        returns_df: 日收益率 DataFrame (columns=资产)

    返回:
        factors: DataFrame, 每列一个资产的趋势因子
        scores: DataFrame, 趋势评分
    """
    factors = pd.DataFrame(1.0, index=returns_df.index, columns=returns_df.columns)
    scores = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) < long * 5:
            continue

        # 滚动计算
        for t in range(long * 5, len(series)):
            window = series.iloc[:t+1]
            factor, score, _ = compute_trend_factor(window, short, medium, long)
            factors.loc[series.index[t], col] = factor
            scores.loc[series.index[t], col] = score

    return factors, scores


def compute_trend_factor_fast(returns_df, short=13, medium=26, long=52):
    """快速版: 只用最近一天的趋势因子 (用于每日回测).

    参数:
        returns_df: 日收益率 DataFrame

    返回:
        factors: Series, 每个资产的趋势因子 (最新值)
    """
    factors = {}
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) < long * 5:
            factors[col] = 1.0
            continue

        factor, _, _ = compute_trend_factor(series, short, medium, long)
        factors[col] = factor

    return pd.Series(factors)
