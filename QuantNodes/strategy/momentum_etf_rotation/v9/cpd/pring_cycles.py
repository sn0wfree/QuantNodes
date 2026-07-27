# coding=utf-8
"""Pring 多周期理论.

Pring 周期理论 (Martin Pring, 2002):
    超长波 Kondratieff:  50-54 年
    长波   Kuznets:      15-25 年
    中波   Juglar:        7-11 年
    短波   Kitchin:       3-5 年
    超短波 Minor:         9-12 月

Pring 10 年周期 (Decennial Pattern):
    第 1-3 年: 熊市主导
    第 4-6 年: 牛市启动
    第 7-9 年: 牛市顶部
    第 10 年: 顶部/调整

中国市场校准:
    基准 2015 年 = 第 7 年 (6124 大顶)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PRING_SEASONALITY = {
    1: "bear_dominant",
    2: "bear_dominant",
    3: "bear_dominant",
    4: "bull_start",
    5: "bull_start",
    6: "bull_start",
    7: "bull_top",
    8: "bull_top",
    9: "bull_top",
    10: "adjustment",
}

PRING_SEASONALITY_CN = {
    1: "熊市主导",
    2: "熊市主导",
    3: "熊市主导",
    4: "牛市启动",
    5: "牛市启动",
    6: "牛市启动",
    7: "牛市顶部",
    8: "牛市顶部",
    9: "牛市顶部",
    10: "顶部调整",
}


def pring_decennial_position(
    year: int,
    base_year: int = 2015,
    base_position: int = 7,
) -> int:
    """Pring 10 年周期位置.

    参数:
        year: 当前年份
        base_year: 基准年份 (默认 2015)
        base_position: 基准年份在 10 年周期中的位置 (默认 7)

    返回:
        1-10, 在 10 年周期中的位置
    """
    diff = year - base_year
    pos = ((base_position - 1 + diff) % 10) + 1
    return pos


def pring_decennial_seasonality(
    year: int,
    base_year: int = 2015,
    base_position: int = 7,
) -> dict:
    """Pring 10 年周期季节性判定.

    返回:
        dict: {position, seasonality_en, seasonality_cn, expected_pattern}
    """
    pos = pring_decennial_position(year, base_year, base_position)
    return {
        "year": year,
        "position": pos,
        "seasonality_en": PRING_SEASONALITY[pos],
        "seasonality_cn": PRING_SEASONALITY_CN[pos],
    }


def _classify_phase_by_slope(series: pd.Series, window: int = 12) -> str:
    """由时间序列斜率判定相位 (up/down/transition).

    参数:
        series: 时间序列
        window: 滑动窗口 (默认 12 周 ≈ 3 月)

    返回:
        'up' / 'down' / 'transition'
    """
    if len(series.dropna()) < window:
        return "transition"

    recent = series.dropna().iloc[-window:]
    slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
    std = recent.std()
    threshold = std / len(recent) * 0.1 if std > 0 else 1e-10

    if slope > threshold:
        return "up"
    elif slope < -threshold:
        return "down"
    else:
        return "transition"


def multi_cycle_position(
    imfs: np.ndarray,
    imf_names: list[str] | None = None,
    date_index: pd.DatetimeIndex | None = None,
    window: int = 12,
) -> pd.DataFrame:
    """多周期相位识别.

    参数:
        imfs: (K, T) 多个 IMF (K 个周期分量, T 时长)
        imf_names: 周期名称列表 (默认 ['Kitchin', 'Juglar', 'Kuznets', 'Kondratieff'])
        date_index: 时间索引
        window: 相位判定窗口

    返回:
        DataFrame columns = 各 IMF 的 phase ('up'/'down'/'transition')
    """
    K, T = imfs.shape
    if imf_names is None:
        imf_names = [f"IMF{i+1}" for i in range(K)]
    if date_index is None:
        date_index = pd.RangeIndex(T)

    assert K == len(imf_names), "IMF 数与名称数不匹配"
    assert T == len(date_index), "IMF 时长与索引不匹配"

    df_dict = {}
    for k in range(K):
        series = pd.Series(imfs[k], index=date_index)
        df_dict[imf_names[k]] = series.rolling(window, min_periods=window).apply(
            lambda x: _classify_phase_by_slope(pd.Series(x), window=window) != "transition",
            raw=True,
        )

    return pd.DataFrame(df_dict, index=date_index)


def multi_cycle_overall(imf_phases: pd.DataFrame) -> str:
    """多周期综合判定.

    参数:
        imf_phases: DataFrame, 各列是 IMF 相位 ('up'=True, 'down'=False)

    返回:
        'STRONG_BULL' / 'MIXED_RECOVERY' / 'TRANSITION' / 'STRONG_BEAR'
    """
    n_up = imf_phases.sum().sum()
    n_total = imf_phases.size
    ratio = n_up / n_total if n_total > 0 else 0.5

    if ratio >= 0.75:
        return "STRONG_BULL"
    elif ratio >= 0.50:
        return "MIXED_RECOVERY"
    elif ratio >= 0.25:
        return "TRANSITION"
    else:
        return "STRONG_BEAR"


def composite_cycle_signal(
    imfs: np.ndarray,
    date_index: pd.DatetimeIndex,
    window: int = 12,
) -> pd.DataFrame:
    """综合周期信号 (up/down 加权).

    参数:
        imfs: (K, T) IMF
        date_index: 时间索引
        window: 滑动窗口

    返回:
        DataFrame columns = [phase, slope, signal]
    """
    K, T = imfs.shape

    slopes = np.zeros(T)
    for t in range(window, T):
        for k in range(K):
            recent = imfs[k, t - window:t]
            if not np.isnan(recent).any():
                slopes[t] += np.polyfit(np.arange(len(recent)), recent, 1)[0]

    avg_slope = slopes / K

    phase = pd.Series("transition", index=date_index, dtype=object)
    phase[avg_slope > 0] = "up"
    phase[avg_slope < 0] = "down"

    signal = (avg_slope > 0).astype(int)

    return pd.DataFrame(
        {"phase": phase, "slope": avg_slope, "signal": signal},
        index=date_index,
    )
