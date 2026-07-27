# coding=utf-8
"""美林时钟 4 阶段识别.

美林时钟 (Merrill Lynch Investment Clock, 2004):
    Phase I - Recovery (复苏):  GDP ↑, CPI ↓  → 股票 > 债券
    Phase II - Overheat (过热): GDP ↑, CPI ↑  → 商品 > 股票
    Phase III - Stagflation (滞胀): GDP ↓, CPI ↑ → 现金 > 商品
    Phase IV - Recession (衰退): GDP ↓, CPI ↓ → 债券 > 股票

中国数据映射:
    GDP 同比: 宏观增长因子 (v7_14_X_panel 第 1 列)
    CPI 同比: 宏观通胀因子_生活端 (第 2 列)
    PPI 同比: 宏观通胀因子_生产端 (第 3 列, 备用)

注: 输入数据为截面标准化的 z-score (相对 43 ETF 横截面均值),
    所以 "上行" 意味着滚动均值 > 0 (高于截面均值).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


MERRILL_PHASE_NAMES = {
    0: "Recovery",
    1: "Overheat",
    2: "Stagflation",
    3: "Recession",
}

MERRILL_PHASE_NAMES_CN = {
    0: "复苏",
    1: "过热",
    2: "滞胀",
    3: "衰退",
}

MERRILL_PHASE_NUM = {"Recovery": 0, "Overheat": 1, "Stagflation": 2, "Recession": 3}


def detect_merrill_phase(
    growth_series: pd.Series,
    inflation_series: pd.Series,
    smooth_window: int = 6,
    threshold_window: int = 36,
) -> pd.Series:
    """美林时钟 4 阶段识别.

    参数:
        growth_series: 经济增长代理 (GDP 同比 / 宏观增长因子)
        inflation_series: 通胀代理 (CPI 同比 / 宏观通胀因子)
        smooth_window: 平滑窗口 (周, 默认 6 周 ≈ 1.5 月)
        threshold_window: 中位数计算窗口 (周, 默认 36 周 ≈ 9 月)

    返回:
        phase: int Series, 0=Recovery, 1=Overheat, 2=Stagflation, 3=Recession
    """
    assert len(growth_series) == len(inflation_series)
    assert (growth_series.index == inflation_series.index).all()

    g = growth_series.copy().astype(float)
    c = inflation_series.copy().astype(float)

    g_smooth = g.rolling(smooth_window, min_periods=1).mean()
    c_smooth = c.rolling(smooth_window, min_periods=1).mean()

    g_threshold = g.rolling(threshold_window, min_periods=threshold_window).median()
    c_threshold = c.rolling(threshold_window, min_periods=threshold_window).median()

    phase = pd.Series(0, index=g.index, dtype=int)

    g_up = g_smooth > g_threshold
    c_up = c_smooth > c_threshold

    phase[g_up & ~c_up] = 0
    phase[g_up & c_up] = 1
    phase[~g_up & c_up] = 2
    phase[~g_up & ~c_up] = 3

    return phase


def detect_merrill_phase_with_confidence(
    growth_series: pd.Series,
    inflation_series: pd.Series,
    smooth_window: int = 6,
    threshold_window: int = 36,
) -> pd.DataFrame:
    """美林时钟识别, 含置信度评分.

    置信度 = 1 - 距边界的相对距离 (标准化).

    返回:
        DataFrame columns = [phase, phase_name, confidence, g_value, c_value]
    """
    phase = detect_merrill_phase(growth_series, inflation_series, smooth_window, threshold_window)

    g = growth_series.rolling(smooth_window, min_periods=1).mean()
    c = inflation_series.rolling(smooth_window, min_periods=1).mean()
    g_med = growth_series.rolling(threshold_window, min_periods=threshold_window).median()
    c_med = inflation_series.rolling(threshold_window, min_periods=threshold_window).median()

    g_std = growth_series.rolling(threshold_window, min_periods=threshold_window).std()
    c_std = inflation_series.rolling(threshold_window, min_periods=threshold_window).std()

    g_z = (g - g_med) / (g_std + 1e-10)
    c_z = (c - c_med) / (c_std + 1e-10)

    confidence = 1 - 1 / (1 + np.maximum(np.abs(g_z), np.abs(c_z)) / 2)
    confidence = confidence.clip(0.0, 1.0).fillna(0.0)

    df = pd.DataFrame(
        {
            "phase": phase,
            "phase_name": phase.map(MERRILL_PHASE_NAMES),
            "phase_name_cn": phase.map(MERRILL_PHASE_NAMES_CN),
            "confidence": confidence,
            "g_value": g,
            "c_value": c,
            "g_zscore": g_z,
            "c_zscore": c_z,
        },
        index=growth_series.index,
    )

    return df


def get_recommended_allocation(phase: int | str) -> dict:
    """基于美林阶段给出推荐资产配置.

    返回:
        dict: {asset: weight}, 权重和 ≈ 1.0
    """
    if isinstance(phase, str):
        phase = MERRILL_PHASE_NUM.get(phase, -1)

    allocation = {
        0: {"股票": 0.60, "债券": 0.30, "商品": 0.05, "现金": 0.05},
        1: {"股票": 0.45, "债券": 0.15, "商品": 0.30, "现金": 0.10},
        2: {"股票": 0.20, "债券": 0.10, "商品": 0.20, "现金": 0.50},
        3: {"股票": 0.25, "债券": 0.55, "商品": 0.05, "现金": 0.15},
    }

    if phase not in allocation:
        return {"股票": 0.40, "债券": 0.30, "商品": 0.15, "现金": 0.15}

    return allocation[phase]


def historical_backtest_check(
    phase_series: pd.Series,
    index_returns: pd.Series,
) -> dict:
    """历史回测验证: 检查美林时钟识别与历史叙事的一致性.

    返回:
        dict: 各阶段对应的平均收益 + 命中率
    """
    common_idx = phase_series.index.intersection(index_returns.index)
    phase_aligned = phase_series.loc[common_idx]
    returns_aligned = index_returns.loc[common_idx]

    result = {}
    for phase_num, phase_name in MERRILL_PHASE_NAMES.items():
        mask = phase_aligned == phase_num
        n_periods = mask.sum()
        mean_ret = returns_aligned[mask].mean() if n_periods > 0 else np.nan
        std_ret = returns_aligned[mask].std() if n_periods > 0 else np.nan
        sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 0 and not np.isnan(std_ret) else np.nan

        result[phase_name] = {
            "n_periods": int(n_periods),
            "mean_weekly_return": float(mean_ret) if not np.isnan(mean_ret) else None,
            "std_weekly_return": float(std_ret) if not np.isnan(std_ret) else None,
            "weekly_sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        }

    return result
