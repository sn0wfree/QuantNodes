# coding=utf-8
"""v9 三维宏观状态分类 — 增长×通胀×流动性 = 8种环境.

三维分类:
    增长: 宏观增长因子 (PMI代理)
    通胀: 宏观通胀因子_生活端 (CPI代理)
    流动性: 信用利差 + 期限利差 (信贷+利率)

8种环境:
    1. 增长↑ 通胀↓ 流动性松 → 强势复苏
    2. 增长↑ 通胀↓ 流动性紧 → 温和复苏
    3. 增长↑ 通胀↑ 流动性松 → 温和扩张
    4. 增长↑ 通胀↑ 流动性紧 → 过热
    5. 增长↓ 通胀↓ 流动性松 → 衰退末期
    6. 增长↓ 通胀↓ 流动性紧 → 深度衰退
    7. 增长↓ 通胀↑ 流动性松 → 弱复苏
    8. 增长↓ 通胀↑ 流动性紧 → 滞胀
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


# 8种环境的配置倾向 (股票/债券/商品/现金)
REGIME_ALLOCATION = {
    'strong_recovery':   {'stock': 0.55, 'bond': 0.15, 'commodity': 0.20, 'cash': 0.10},
    'mild_recovery':     {'stock': 0.45, 'bond': 0.20, 'commodity': 0.15, 'cash': 0.20},
    'mild_expansion':    {'stock': 0.40, 'bond': 0.20, 'commodity': 0.25, 'cash': 0.15},
    'overheat':          {'stock': 0.25, 'bond': 0.15, 'commodity': 0.35, 'cash': 0.25},
    'late_recession':    {'stock': 0.25, 'bond': 0.40, 'commodity': 0.10, 'cash': 0.25},
    'deep_recession':    {'stock': 0.10, 'bond': 0.50, 'commodity': 0.05, 'cash': 0.35},
    'weak_recovery':     {'stock': 0.30, 'bond': 0.30, 'commodity': 0.15, 'cash': 0.25},
    'stagflation':       {'stock': 0.10, 'bond': 0.25, 'commodity': 0.20, 'cash': 0.45},
}

REGIME_NAMES_CN = {
    'strong_recovery': '强势复苏',
    'mild_recovery': '温和复苏',
    'mild_expansion': '温和扩张',
    'overheat': '过热',
    'late_recession': '衰退末期',
    'deep_recession': '深度衰退',
    'weak_recovery': '弱复苏',
    'stagflation': '滞胀',
}


def classify_dimension(series, window=52):
    """判断维度方向 (上行/下行).

    参数:
        series: Z-score 序列
        window: HP 滤波窗口

    返回:
        'up' 或 'down'
    """
    if len(series.dropna()) < window:
        return 'up'

    # HP 滤波提取趋势
    try:
        _, trend = hpfilter(series.dropna(), lamb=100)
        # 判断趋势方向: 最近 26 周斜率
        recent = trend.iloc[-26:]
        slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
        std = trend.std()
        threshold = std / len(trend) * 0.01 if std > 0 else 1e-10
        return 'up' if slope > threshold else 'down'
    except:
        return 'up'


def classify_liquidity(credit_series, term_series, window=52):
    """流动性分类 (宽松/紧缩).

    逻辑:
        信用利差下行 → 宽松 (借钱更容易)
        期限利差上行 → 宽松 (收益率曲线陡峭)

    返回:
        'easy' 或 'tight'
    """
    if len(credit_series.dropna()) < window:
        return 'easy'

    try:
        _, credit_trend = hpfilter(credit_series.dropna(), lamb=100)
        _, term_trend = hpfilter(term_series.dropna(), lamb=100)

        # 信用利差趋势 (反向: 下行=宽松)
        credit_slope = np.polyfit(
            np.arange(26), credit_trend.iloc[-26:].values, 1
        )[0]

        # 期限利差趋势 (正向: 上行=宽松)
        term_slope = np.polyfit(
            np.arange(26), term_trend.iloc[-26:].values, 1
        )[0]

        # 综合判断
        if credit_slope < 0 and term_slope > 0:
            return 'easy'
        elif credit_slope > 0 and term_slope < 0:
            return 'tight'
        elif credit_slope < 0:
            return 'easy'
        else:
            return 'tight'
    except:
        return 'easy'


def classify_regime(growth_dir, inflation_dir, liquidity_dir):
    """三维分类 → 8种环境.

    返回:
        regime_key: 环境标识
        allocation: 配置比例
    """
    if growth_dir == 'up' and inflation_dir == 'down' and liquidity_dir == 'easy':
        return 'strong_recovery'
    elif growth_dir == 'up' and inflation_dir == 'down' and liquidity_dir == 'tight':
        return 'mild_recovery'
    elif growth_dir == 'up' and inflation_dir == 'up' and liquidity_dir == 'easy':
        return 'mild_expansion'
    elif growth_dir == 'up' and inflation_dir == 'up' and liquidity_dir == 'tight':
        return 'overheat'
    elif growth_dir == 'down' and inflation_dir == 'down' and liquidity_dir == 'easy':
        return 'late_recession'
    elif growth_dir == 'down' and inflation_dir == 'down' and liquidity_dir == 'tight':
        return 'deep_recession'
    elif growth_dir == 'down' and inflation_dir == 'up' and liquidity_dir == 'easy':
        return 'weak_recovery'
    else:  # down, up, tight
        return 'stagflation'


def detect_macro_regime(macro_df):
    """宏观状态分类主入口.

    参数:
        macro_df: 宏观因子 DataFrame (columns: 宏观增长因子, 宏观通胀因子_生活端, 信用利差因子, 期限利差因子_债)

    返回:
        DataFrame: columns = [regime, allocation, growth_dir, inflation_dir, liquidity_dir]
    """
    growth = macro_df['宏观增长因子'].dropna()
    inflation = macro_df['宏观通胀因子_生活端'].dropna()
    credit = macro_df['信用利差因子'].dropna()
    term = macro_df['期限利差因子_债'].dropna()

    # 对齐索引
    common_idx = growth.index.intersection(inflation.index).intersection(credit.index).intersection(term.index)
    growth = growth.loc[common_idx]
    inflation = inflation.loc[common_idx]
    credit = credit.loc[common_idx]
    term = term.loc[common_idx]

    # Z-score
    growth_z = (growth - growth.mean()) / (growth.std() + 1e-10)
    inflation_z = (inflation - inflation.mean()) / (inflation.std() + 1e-10)
    credit_z = (credit - credit.mean()) / (credit.std() + 1e-10)
    term_z = (term - term.mean()) / (term.std() + 1e-10)

    # 逐周分类 (用滚动窗口)
    regimes = []
    allocations = []
    growth_dirs = []
    inflation_dirs = []
    liquidity_dirs = []

    window = 52
    for t in range(len(common_idx)):
        if t < window:
            regimes.append('mild_recovery')
            allocations.append(REGIME_ALLOCATION['mild_recovery'])
            growth_dirs.append('up')
            inflation_dirs.append('down')
            liquidity_dirs.append('easy')
            continue

        g_dir = classify_dimension(growth_z.iloc[:t+1])
        i_dir = classify_dimension(inflation_z.iloc[:t+1])
        l_dir = classify_liquidity(credit_z.iloc[:t+1], term_z.iloc[:t+1])

        regime = classify_regime(g_dir, i_dir, l_dir)
        regimes.append(regime)
        allocations.append(REGIME_ALLOCATION[regime])
        growth_dirs.append(g_dir)
        inflation_dirs.append(i_dir)
        liquidity_dirs.append(l_dir)

    # 构建结果
    alloc_df = pd.DataFrame(allocations, index=common_idx)
    result = pd.DataFrame({
        'regime': regimes,
        'growth_dir': growth_dirs,
        'inflation_dir': inflation_dirs,
        'liquidity_dir': liquidity_dirs,
    }, index=common_idx)

    return result, alloc_df
