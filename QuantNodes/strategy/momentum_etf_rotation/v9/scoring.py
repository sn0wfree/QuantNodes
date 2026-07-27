# coding=utf-8
"""v9 评分合成模块 (40+40+20).

评分维度:
    1. 周期趋势 (cycle):     0-40, 基于 4 IMF 的方向加权
    2. 周期耦合 (coupling):   0-40, 基于 Hilbert + 双相干
    3. VIX (vix):            0-20, 基于美股 VIX 倒数百分位

总分: 0-100, 阈值 50/30 (迟滞信号).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_IMF_WEIGHTS = {"IMF1": 0.10, "IMF2": 0.20, "IMF3": 0.40, "IMF4": 0.30}


def compute_cycle_score(
    imfs: np.ndarray,
    window: int = 12,
    weights: dict | None = None,
) -> tuple:
    """计算周期趋势评分时序 (0-40).

    评分规则:
        - 每个 IMF 上行 → 10 分
        - 4 个 IMF 满分 = 40 分
        - weights: 可选加权 (默认等权 10 分/IMF)

    参数:
        imfs: (K, T) IMF 数组
        window: 滑动窗口 (默认 12 周)
        weights: {imf_name: weight}, 默认 DEFAULT_IMF_WEIGHTS

    返回:
        score: (T,) 评分时序 (0-40)
        imf_slopes: (K, T) 各 IMF 斜率
        multi_phases: (K, T) bool, True=上行
    """
    if weights is None:
        weights = DEFAULT_IMF_WEIGHTS

    K, T = imfs.shape

    if T < window:
        return (
            pd.Series(0.0, index=range(T)),
            np.zeros((K, T)),
            np.zeros((K, T), dtype=bool),
        )

    imf_slopes = np.zeros((K, T))
    multi_phases = np.zeros((K, T), dtype=bool)
    score = np.zeros(T)

    for t in range(window, T):
        ts = np.arange(window, dtype=float)
        for k in range(K):
            y = imfs[k, t - window:t]
            slope = np.polyfit(ts, y, 1)[0]
            imf_slopes[k, t] = slope
            if slope > 0:
                multi_phases[k, t] = True

        s = 0.0
        for k in range(K):
            name = f"IMF{k+1}"
            w = weights.get(name, 1.0)
            if multi_phases[k, t]:
                s += 10.0 * w
        score[t] = min(s, 40.0)

    return pd.Series(score), imf_slopes, multi_phases


def compute_coupling_score_timeseries(
    coupling_result: dict,
    max_pairs: int = 4,
    bic_threshold: float = 0.6,
) -> pd.Series:
    """计算耦合评分时序 (封装 coupling.py).

    参数:
        coupling_result: coupling_analysis() 输出
        max_pairs: 最多参与评分的 IMF 对数
        bic_threshold: 双相干显著阈值

    返回:
        score: 评分时序 (0-40)
    """
    from .coupling import compute_coupling_score_timeseries as _impl

    return _impl(coupling_result, max_pairs=max_pairs, bic_threshold=bic_threshold)


def compute_vix_score(
    vix_series: pd.Series,
    window: int = 252,
    cn_calibration: bool = True,
) -> pd.Series:
    """VIX 评分 0-20.

    A 股校准 (基于美股 VIX 作为全球风险偏好代理):
        - VIX < 15: 20 分
        - VIX 15-20: 15 分
        - VIX 20-25: 12 分
        - VIX 25-30: 8 分
        - VIX 30-35: 4 分
        - VIX >= 35: 0 分

    参数:
        vix_series: VIX 时序
        window: 滚动窗口 (默认 252)
        cn_calibration: 是否使用 A 股校准 (默认 True)

    返回:
        score: 评分时序 (0-20)
    """
    if cn_calibration:
        conditions = [15, 20, 25, 30, 35]
        scores = [20, 15, 12, 8, 4]
        result = pd.Series(0.0, index=vix_series.index)
        result[vix_series < 15] = 20
        for cond, s in zip(conditions, scores):
            result[(vix_series >= cond) & (vix_series < cond + 5)] = s
        return result
    else:
        inv_vix = 1 / vix_series
        pct = inv_vix.rolling(window, min_periods=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        return (pct * 20).fillna(10).clip(0, 20)


def compute_total_score(
    cycle_score: pd.Series,
    coupling_score: pd.Series,
    vix_score: pd.Series,
) -> pd.Series:
    """总分 = 周期趋势 + 周期耦合 + VIX (0-100)."""
    return (cycle_score + coupling_score + vix_score).clip(0, 100)


def score_to_signal_hysteresis(
    score_series: pd.Series,
    upper: float = 50.0,
    lower: float = 30.0,
    initial: int = 0,
) -> pd.Series:
    """迟滞信号生成 (防抖).

    状态机:
        - 当前 0, score >= upper → 1
        - 当前 1, score <= lower → 0
        - 其他 → 保持上一状态

    参数:
        score_series: 评分时序
        upper: 上阈值 (默认 50)
        lower: 下阈值 (默认 30)
        initial: 初始信号

    返回:
        signal: 时序 (0 或 1)
    """
    signals = np.zeros(len(score_series), dtype=int)
    current = initial

    for i, s in enumerate(score_series):
        if pd.isna(s):
            signals[i] = current
            continue

        if current == 0 and s >= upper:
            current = 1
        elif current == 1 and s <= lower:
            current = 0
        signals[i] = current

    return pd.Series(signals, index=score_series.index)


def score_to_simple_signal(
    score_series: pd.Series,
    upper: float = 50.0,
    lower: float = 30.0,
) -> pd.Series:
    """简单信号 (无迟滞).

    score >= upper → 1
    score <= lower → 0
    其他 → NaN
    """
    result = pd.Series(np.nan, index=score_series.index)
    result[score_series >= upper] = 1
    result[score_series <= lower] = 0
    return result


def compute_score_timeseries(
    imfs: np.ndarray,
    coupling_result: dict,
    vix_series: pd.Series,
    date_index: pd.DatetimeIndex,
    window: int = 12,
    imf_weights: dict | None = None,
    vix_window: int = 252,
    cn_calibration: bool = True,
) -> tuple:
    """主入口: 计算完整评分时序.

    参数:
        imfs: (K, T) IMF 数组
        coupling_result: coupling_analysis() 输出
        vix_series: VIX 时序
        date_index: 时间索引
        window: 周期评分窗口
        imf_weights: IMF 权重
        vix_window: VIX 评分窗口
        cn_calibration: A 股校准

    返回:
        total_score, cycle_score, coupling_score, vix_score, multi_phases, signal
    """
    cycle_score, imf_slopes, multi_phases = compute_cycle_score(
        imfs, window=window, weights=imf_weights
    )
    cycle_score = cycle_score.reindex(date_index).fillna(0.0)

    coupling_score = compute_coupling_score_timeseries(coupling_result)
    coupling_score = coupling_score.reindex(date_index).fillna(0.0)

    vix_aligned = vix_series.reindex(date_index, method="ffill")
    vix_score = compute_vix_score(vix_aligned, window=vix_window, cn_calibration=cn_calibration).fillna(10.0)

    total_score = compute_total_score(cycle_score, coupling_score, vix_score).fillna(0.0)

    signal = score_to_signal_hysteresis(total_score, upper=50.0, lower=30.0, initial=1)

    return total_score, cycle_score, coupling_score, vix_score, multi_phases, signal
