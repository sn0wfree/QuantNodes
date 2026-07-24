# coding=utf-8
"""v9 周期耦合识别 (Hilbert 相位 + 双相干).

Phase 2 核心模块:
    1. Hilbert 变换提取瞬时相位
    2. 相位锁定值 (PLV) 计算
    3. 双相干系数 (Bicoherence) 计算
    4. 耦合信号合成 → 0-40 评分
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import hilbert


def compute_instantaneous_phase(imf: np.ndarray, unwrap: bool = False) -> np.ndarray:
    """由 IMF 计算瞬时相位.

    参数:
        imf: 1-D IMF 信号
        unwrap: 是否解缠绕 (单调递增)

    返回:
        phase: 瞬时相位 ∈ [-π, π] (或解缠绕后单调递增)
    """
    analytic_signal = hilbert(imf)
    phase = np.angle(analytic_signal)
    if unwrap:
        phase = np.unwrap(phase)
    return phase


def compute_all_phases(imfs: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """对所有 IMF 计算瞬时相位.

    参数:
        imfs: (K, T) IMF 数组
        unwrap: 是否解缠绕

    返回:
        phases: (K, T) 相位数组
    """
    K = imfs.shape[0]
    phases = np.zeros_like(imfs)
    for k in range(K):
        phases[k] = compute_instantaneous_phase(imfs[k], unwrap=unwrap)
    return phases


def compute_delta_phases(phases: np.ndarray, imf_names: list[str] | None = None) -> dict:
    """计算所有 IMF 对的相位差.

    参数:
        phases: (K, T) 相位数组
        imf_names: IMF 名称 (默认 IMF1, IMF2, ...)

    返回:
        {(imf_i, imf_j): delta_phase (T,)}
    """
    K = phases.shape[0]
    if imf_names is None:
        imf_names = [f"IMF{i+1}" for i in range(K)]

    delta_phases = {}
    for i, j in combinations(range(K), 2):
        delta = phases[j] - phases[i]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        delta_phases[(imf_names[i], imf_names[j])] = delta

    return delta_phases


def compute_phase_locking_value(
    delta_phases: np.ndarray | dict,
    window: int = 12,
) -> pd.Series | dict:
    """滚动 PLV (Phase Locking Value).

    PLV ∈ [0, 1], 越接近 1 表示相位锁定越强.

    参数:
        delta_phases: 相位差数组 (T,) 或字典 {(pair): delta_phase}
        window: 滚动窗口 (默认 12 周 ≈ 3 月)

    返回:
        PLV 时序 (pd.Series) 或 {pair: PLV 时序}
    """
    if isinstance(delta_phases, dict):
        return {pair: compute_phase_locking_value(dp, window) for pair, dp in delta_phases.items()}

    def plv_window(x):
        return np.abs(np.mean(np.exp(1j * x)))

    series = pd.Series(delta_phases)
    return series.rolling(window, min_periods=window).apply(plv_window, raw=True)


def detect_phase_locking(
    delta_phase: np.ndarray,
    threshold_deg: float = 30.0,
    min_duration: int = 12,
) -> np.ndarray:
    """检测相位锁定时段.

    使用滑动窗口 (而非要求连续) 来判定锁定, 更稳健.

    参数:
        delta_phase: 相位差 (T,)
        threshold_deg: 锁定阈值 (度, 默认 30°)
        min_duration: 最短持续期 (默认 12 周, 用于累积窗口)

    返回:
        locked: (T,) bool 数组, True 表示锁定
    """
    threshold_rad = np.deg2rad(threshold_deg)
    locked_raw = np.abs(delta_phase) < threshold_rad

    locked = pd.Series(locked_raw.astype(int)).rolling(min_duration, min_periods=min_duration).mean()
    locked = (locked >= 0.6).fillna(False).values.astype(bool)

    return locked


def bicoherence(
    signal: np.ndarray,
    nperseg: int | None = None,
    noverlap: int | None = None,
    fs: float = 1.0,
) -> tuple:
    """双相干系数计算.

    参数:
        signal: 1-D 输入信号
        nperseg: 每段长度 (默认 len(signal)//8)
        noverlap: 段重叠 (默认 nperseg//2)
        fs: 采样率 (周频=1)

    返回:
        freq: 频率数组
        bic: 双相干矩阵 (n_freq, n_freq)
        p_value: 显著性 p 值矩阵
    """
    N = len(signal)
    if nperseg is None:
        nperseg = min(64, N // 4)
    if noverlap is None:
        noverlap = nperseg // 2

    from scipy.signal import spectrogram

    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    n_freq = len(f)
    n_seg = Sxx.shape[1]

    bic = np.zeros((n_freq, n_freq))
    p_value = np.ones((n_freq, n_freq))

    for i in range(n_freq):
        for j in range(i, n_freq):
            if i + j < n_freq:
                B = np.mean(Sxx[i] * Sxx[j] * np.conj(Sxx[i + j]))
                norm = np.sqrt(
                    np.mean(np.abs(Sxx[i] * Sxx[j]) ** 2) * np.mean(np.abs(Sxx[i + j]) ** 2)
                )
                if norm > 0:
                    bic[i, j] = np.abs(B) / norm
                    bic[j, i] = bic[i, j]
                    chi2_stat = 2 * n_seg * bic[i, j] ** 2
                    from scipy.stats import chi2

                    p_value[i, j] = 1 - chi2.cdf(chi2_stat, df=2)
                    p_value[j, i] = p_value[i, j]

    return f, bic, p_value


def coupling_analysis(
    imfs: np.ndarray,
    date_index: pd.DatetimeIndex | None = None,
    lock_threshold_deg: float = 30.0,
    lock_min_duration: int = 12,
    plv_window: int = 12,
) -> dict:
    """综合耦合分析.

    参数:
        imfs: (K, T) IMF 数组
        date_index: 时间索引
        lock_threshold_deg: 相位锁定阈值
        lock_min_duration: 锁定最短持续期
        plv_window: PLV 计算窗口

    返回:
        dict: {
            'phases': (K, T) 相位数组,
            'delta_phases': {pair: delta_phase},
            'plv': {pair: PLV 时序},
            'locked': {pair: locked 时序},
            'n_locked_pairs': (T,) 锁定对数,
            'bic_freq', 'bic_matrix', 'bic_pvalue': 双相干结果,
            'bic_max': 最大双相干
        }
    """
    K, T = imfs.shape
    if date_index is None:
        date_index = pd.RangeIndex(T)
    elif len(date_index) != T:
        date_index = date_index[:T]

    imf_names = [f"IMF{i+1}" for i in range(K)]

    phases = compute_all_phases(imfs, unwrap=True)
    delta_phases = compute_delta_phases(phases, imf_names)
    plv = compute_phase_locking_value(delta_phases, window=plv_window)

    locked = {}
    for pair, dp in delta_phases.items():
        locked[pair] = detect_phase_locking(
            dp, threshold_deg=lock_threshold_deg, min_duration=lock_min_duration
        )

    n_locked_pairs = pd.Series(0, index=date_index, dtype=int)
    for pair, lk in locked.items():
        n_locked_pairs = n_locked_pairs.add(pd.Series(lk.astype(int), index=date_index), fill_value=0)

    hs300_proxy = imfs.sum(axis=0)
    bic_freq, bic_matrix, bic_pvalue = bicoherence(hs300_proxy, fs=1.0)
    bic_max = float(np.nanmax(bic_matrix))

    n_locked_pairs = n_locked_pairs.reindex(date_index)

    return {
        "phases": phases,
        "delta_phases": delta_phases,
        "plv": plv,
        "locked": locked,
        "n_locked_pairs": n_locked_pairs,
        "bic_freq": bic_freq,
        "bic_matrix": bic_matrix,
        "bic_pvalue": bic_pvalue,
        "bic_max": bic_max,
        "imf_names": imf_names,
    }


def compute_coupling_score_timeseries(
    coupling_result: dict,
    max_pairs: int = 4,
    bic_threshold: float = 0.6,
) -> pd.Series:
    """计算耦合评分时序.

    评分规则 (0-40):
        - 每对 IMF 相位锁定 (Δφ < 30° 持续 3 月): +10 分
        - 双相干显著 (max bic > 0.6): +10 分 (一次性奖励)

    参数:
        coupling_result: coupling_analysis() 输出
        max_pairs: 最多参与评分的 IMF 对数
        bic_threshold: 双相干显著阈值

    返回:
        score: 评分时序 (0-40)
    """
    n_locked = coupling_result["n_locked_pairs"]
    bic_max = coupling_result["bic_max"]

    score = n_locked.clip(upper=max_pairs) * 10.0

    bic_bonus = 10.0 if bic_max > bic_threshold else 0.0
    score = score + bic_bonus

    return score.clip(upper=40.0)