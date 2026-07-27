# coding=utf-8
"""v9 仓位合成与大盘信号.

v9 信号层级:
    final_weight(t, etf) = v9_signal(t) × v7_weight(t, etf) × v8_factor(t, etf)

v9 大盘信号 0/1 决定是否参与市场.
v7.14 ETF 权重决定选什么.
v8 Bear% 决定仓位多少.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]


def v8_bear_to_factor(
    bear_pct: float | pd.Series,
    threshold: float = 0.3,
) -> float | pd.Series:
    """v8 Bear% → 仓位因子.

    参数:
        bear_pct: Bear% ∈ [0, 1] (单值或 Series)
        threshold: 阈值 (默认 0.3)

    返回:
        factor ∈ [0, 1] (单值或 Series)
    """
    if bear_pct <= threshold:
        return 1.0
    else:
        return np.maximum(1 - (bear_pct - threshold) / (1 - threshold), 0)


def align_v9_to_weekly(
    v9_signal_weekly: pd.Series,
    weekly_dates: pd.DatetimeIndex,
) -> pd.Series:
    """对齐 v9 信号到周频日期.

    参数:
        v9_signal_weekly: 周频 v9 信号
        weekly_dates: 目标周频日期

    返回:
        对齐后的 v9 信号
    """
    return v9_signal_weekly.reindex(weekly_dates, method="ffill").fillna(0)


def synthesize_position(
    v9_signal: pd.Series,
    v7_weights: pd.DataFrame,
    v8_factors: pd.DataFrame,
) -> pd.DataFrame:
    """三层仓位合成.

    参数:
        v9_signal: (T_weekly,) 大盘信号 0/1
        v7_weights: (T_weekly, N_etf) v7.14 周权重
        v8_factors: (T_weekly, N_etf) v8 仓位因子

    返回:
        final_weights: (T_weekly, N_etf) 合成仓位
    """
    assert len(v9_signal) == len(v7_weights) == len(v8_factors), "三层长度不匹配"
    assert (v9_signal.index == v7_weights.index).all(), "索引不匹配"

    final = v7_weights.copy()
    for date in v7_weights.index:
        if v9_signal.loc[date] == 0:
            final.loc[date] = 0
        else:
            final.loc[date] = v7_weights.loc[date] * v8_factors.loc[date]

    return final


def normalize_weights(final_weights: pd.DataFrame) -> tuple:
    """归一化权重 + 计算现金比例.

    参数:
        final_weights: 合成仓位

    返回:
        normalized: 归一化后的权重 (sum = 1)
        cash: 现金权重 (1 - sum)
    """
    final_weights = final_weights.clip(lower=0)
    row_sums = final_weights.sum(axis=1).replace(0, 1)
    normalized = final_weights.div(row_sums, axis=0)
    cash = 1 - final_weights.div(row_sums, axis=0).sum(axis=1)
    return normalized, cash


def generate_v8_bear_pct_factors(
    daily_returns: pd.DataFrame,
    resample_freq: str = "W",
    bear_window: int = 60,
) -> pd.DataFrame:
    """生成 v8 Bear% 仓位因子.

    简化版: 用滚动波动率作为 Bear% 代理 (不调用 Jump Model, 加快速度).

    参数:
        daily_returns: (T_daily, N_etf) 日频收益
        resample_freq: 重采样频率
        bear_window: Bear% 滚动窗口

    返回:
        factors: (T_weekly, N_etf) v8 仓位因子
    """
    weekly_vol = daily_returns.rolling(20).std().resample(resample_freq).last()

    rolling_corr = weekly_vol.rolling(bear_window).rank(pct=True)

    bear_pct = rolling_corr.clip(0, 1).fillna(0.5)

    factors = bear_pct.apply(v8_bear_to_factor)

    return factors


def simple_etf_equal_weights(
    weekly_dates: pd.DatetimeIndex,
    n_etfs: int = 43,
) -> pd.DataFrame:
    """生成 ETF 等权基础权重 (作为 v7.14 fallback).

    参数:
        weekly_dates: 周频日期
        n_etfs: ETF 数量

    返回:
        weights: (T, N) 等权 DataFrame
    """
    weights = pd.DataFrame(
        1.0 / n_etfs,
        index=weekly_dates,
        columns=[f"ETF{i}" for i in range(n_etfs)],
    )
    return weights


def compute_v9_only_position(
    v9_signal: pd.Series,
    etf_returns: pd.DataFrame,
    signal_threshold: float = 0.5,
) -> pd.DataFrame:
    """v9 单层仓位合成 (不依赖 v7.14/v8).

    当 v9=1 时, 等权持仓所有 ETF.
    当 v9=0 时, 全空仓.

    参数:
        v9_signal: (T,) v9 大盘信号 (0 或 1)
        etf_returns: (T, N) ETF 周收益 (用于确定 ETF 名称)
        signal_threshold: 信号阈值

    返回:
        weights: (T, N) 等权持仓
    """
    binary_signal = (v9_signal >= signal_threshold).astype(int)
    n = etf_returns.shape[1]
    weights = pd.DataFrame(0.0, index=v9_signal.index, columns=etf_returns.columns)
    for date in v9_signal.index:
        if binary_signal.loc[date] == 1:
            weights.loc[date] = 1.0 / n
    return weights
