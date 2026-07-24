# coding=utf-8
"""v9 周期提取模块 — 相位 + 速度 + 加速度.

输出:
    φ(t): 瞬时相位 (Hilbert)
    dφ/dt: 相位速度 (一阶导数)
    d²φ/dt²: 相位加速度 (二阶导数)
    振幅: 周期强度
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt


def bandpass_filter(signal, low_weeks, high_weeks, fs=1.0, order=3):
    """带通滤波."""
    nyq = 0.5 * fs
    low = max((1.0 / high_weeks) / nyq, 0.001)
    high = min((1.0 / low_weeks) / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_cycle(signal, period_low, period_high, fs=1.0):
    """提取周期分量 → 相位 + 速度 + 加速度.

    参数:
        signal: 1-D 信号 (已标准化)
        period_low: 周期下限 (周)
        period_high: 周期上限 (周)
        fs: 采样率 (周频=1)

    返回:
        dict: phase, velocity, acceleration, amplitude, filtered
    """
    # 带通滤波
    filtered = bandpass_filter(signal, period_low, period_high, fs)

    # Hilbert 变换 → 瞬时相位
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)

    # 相位解缠绕 (避免 2π 跳跃)
    phase_unwrapped = np.unwrap(phase)

    # 速度: 相位一阶导数 (差分)
    velocity = np.gradient(phase_unwrapped)

    # 加速度: 相位二阶导数
    acceleration = np.gradient(velocity)

    return {
        'phase': phase_unwrapped,
        'velocity': velocity,
        'acceleration': acceleration,
        'amplitude': amplitude,
        'filtered': filtered,
    }


def extract_multi_scale_cycles(signal, fs=1.0):
    """多尺度周期提取 (Kitchin + Juglar + 长期).

    返回:
        dict: {
            'kitchin': {...},   # Kitchin (100-260周)
            'juglar': {...},    # Juglar (260-520周)
            'long_term': {...}, # 长期趋势 (520+周)
        }
    """
    return {
        'kitchin': extract_cycle(signal, 100, 260, fs),
        'juglar': extract_cycle(signal, 260, 520, fs),
        'long_term': extract_cycle(signal, 520, 2000, fs),
    }


def compute_cycle_strength(cycles):
    """多周期共振强度.

    返回:
        strength: [-1, 1], 正=共振上行, 负=共振下行
    """
    k_phase = cycles['kitchin']['phase']
    j_phase = cycles['juglar']['phase']
    l_phase = cycles['long_term']['phase']

    # 取最近一天的相位
    k_cos = np.cos(k_phase[-1])
    j_cos = np.cos(j_phase[-1])
    l_cos = np.cos(l_phase[-1])

    return (k_cos + j_cos + l_cos) / 3.0


def macro_cycle_extract(macro_df, date_index=None):
    """对宏观指标做周期提取.

    参数:
        macro_df: 宏观因子 DataFrame (index=时间)
        date_index: 输出索引 (默认与 macro_df 相同)

    返回:
        dict: {factor_name: {kitchin: ..., juglar: ..., long_term: ...}}
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    results = {}

    for col in macro_df.columns:
        series = macro_df[col].dropna()
        if len(series) < 100:
            continue

        # Z-score 标准化
        z = (series - series.mean()) / (series.std() + 1e-10)

        # HP 滤波提取周期
        try:
            cycle_hp, trend_hp = hpfilter(z, lamb=100)
        except:
            continue

        # 带通滤波 + Hilbert
        cycles = extract_multi_scale_cycles(cycle_hp.values)

        # 转为 Series
        for scale in ['kitchin', 'juglar', 'long_term']:
            for key in ['phase', 'velocity', 'acceleration', 'amplitude']:
                cycles[scale][key] = pd.Series(
                    cycles[scale][key], index=cycle_hp.index
                )

        results[col] = {
            'cycles': cycles,
            'trend': trend_hp,
            'cycle': cycle_hp,
        }

    return results


def asset_cycle_extract(returns_df, date_index=None):
    """对资产收益率做周期提取 (动量周期).

    参数:
        returns_df: 资产收益率 DataFrame (日频)
        date_index: 输出索引

    返回:
        dict: {asset_name: {kitchin: ..., juglar: ...}}
    """
    results = {}

    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) < 100:
            continue

        # 累积净值
        nav = (1 + series).cumprod()

        # HP 滤波
        from statsmodels.tsa.filters.hp_filter import hpfilter
        try:
            cycle_hp, trend_hp = hpfilter(np.log(nav), lamb=100)
        except:
            continue

        # 周期提取
        cycles = extract_multi_scale_cycles(cycle_hp.values)

        for scale in ['kitchin', 'juglar', 'long_term']:
            for key in ['phase', 'velocity', 'acceleration', 'amplitude']:
                cycles[scale][key] = pd.Series(
                    cycles[scale][key], index=cycle_hp.index
                )

        results[col] = {
            'cycles': cycles,
            'trend': trend_hp,
        }

    return results
