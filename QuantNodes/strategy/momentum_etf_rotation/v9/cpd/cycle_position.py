# coding=utf-8
"""Cycle State 数据类与综合诊断.

CycleState 整合美林时钟 + Pring 周期 + 多周期相位 + 评分, 形成完整的周期状态视图.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .merrill_clock import (
    detect_merrill_phase_with_confidence,
    get_recommended_allocation,
)
from .pring_cycles import (
    pring_decennial_position,
    pring_decennial_seasonality,
)


@dataclass
class CycleState:
    """周期状态综合数据结构.

    整合所有周期诊断结果, 用于报告生成与仪表盘展示.
    """

    merrill_phase: str
    merrill_phase_cn: str
    merrill_phase_num: int
    merrill_confidence: float
    growth_zscore: float
    inflation_zscore: float

    pring_year: int
    pring_position: int
    pring_seasonality: str
    pring_seasonality_cn: str

    multi_cycle_phases: dict
    composite_phase: str

    cycle_score: float
    coupling_score: float
    vix_score: float
    total_score: float

    v9_signal: int
    signal_label: str

    recommended_allocation: dict

    gdp_proxy_name: str
    cpi_proxy_name: str
    vix_value: Optional[float] = None

    report_date: Optional[pd.Timestamp] = None
    data_through: Optional[pd.Timestamp] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_vix_score_cn(vix_value: float) -> float:
    """A 股校准的 VIX 评分 (基于美股 VIX 作为全球风险偏好代理).

    注: v7_14_macro 用的是美股 VIX, 不是中国 VIX (中国 iVIX 数据不完整).
        这里把美股 VIX 作为全球风险偏好代理, 在中国市场上仍有一定预测力.
    """
    if vix_value is None or np.isnan(vix_value):
        return 10.0
    if vix_value < 15:
        return 20.0
    elif vix_value < 20:
        return 15.0
    elif vix_value < 25:
        return 12.0
    elif vix_value < 30:
        return 8.0
    elif vix_value < 35:
        return 4.0
    else:
        return 0.0


def compute_cycle_score(imf_slopes: dict, weights: dict | None = None) -> float:
    """周期趋势评分 0-40.

    参数:
        imf_slopes: {imf_name: slope_value}
        weights: {imf_name: weight}, 默认 {IMF1: 0.10, IMF2: 0.20, IMF3: 0.40, IMF4: 0.30}
    """
    if weights is None:
        weights = {"IMF1": 0.10, "IMF2": 0.20, "IMF3": 0.40, "IMF4": 0.30}

    score = 0.0
    for name, slope in imf_slopes.items():
        w = weights.get(name, 0.25)
        if slope > 0:
            score += 10.0 * w

    return min(score, 40.0)


def compute_coupling_score(locked_pairs: int, bic_max: float, max_pairs: int = 4) -> float:
    """周期耦合评分 0-40.

    参数:
        locked_pairs: 相位锁定对数
        bic_max: 最大双相干系数
        max_pairs: 最多参与评分的 IMF 对数
    """
    lock_score = min(locked_pairs, max_pairs) * 10.0
    bic_bonus = 10.0 if bic_max > 0.6 else 0.0

    return min(lock_score + bic_bonus, 40.0)


def diagnose_current_state(
    growth_series: pd.Series,
    inflation_series: pd.Series,
    vix_series: pd.Series,
    imfs: np.ndarray | None = None,
    locked_pairs: int = 0,
    bic_max: float = 0.0,
    imf_weights: dict | None = None,
    smooth_window: int = 6,
    threshold_window: int = 36,
    data_through: pd.Timestamp | None = None,
) -> CycleState:
    """综合诊断当前周期状态.

    参数:
        growth_series: 经济增长代理 (z-score)
        inflation_series: 通胀代理 (z-score)
        vix_series: VIX 时序 (实际值, 非 z-score)
        imfs: (K, T) 多周期 IMF, 可选
        locked_pairs: 相位锁定对数
        bic_max: 最大双相干
        imf_weights: IMF 权重
        smooth_window: 平滑窗口
        threshold_window: 中位数窗口
        data_through: 数据截至日期

    返回:
        CycleState
    """
    if imf_weights is None:
        imf_weights = {"IMF1": 0.10, "IMF2": 0.20, "IMF3": 0.40, "IMF4": 0.30}

    growth_series = growth_series.dropna()
    inflation_series = inflation_series.dropna()
    vix_series = vix_series.dropna()

    common_idx = growth_series.index.intersection(inflation_series.index)
    growth_series = growth_series.loc[common_idx]
    inflation_series = inflation_series.loc[common_idx]

    merrill_df = detect_merrill_phase_with_confidence(
        growth_series,
        inflation_series,
        smooth_window,
        threshold_window,
    )

    latest = merrill_df.iloc[-1]
    merrill_phase = latest["phase_name"]
    merrill_phase_cn = latest["phase_name_cn"]
    merrill_phase_num = int(latest["phase"])
    merrill_confidence = float(latest["confidence"])
    g_zscore = float(latest["g_zscore"])
    c_zscore = float(latest["c_zscore"])

    current_year = latest.name.year if hasattr(latest.name, "year") else pd.Timestamp.now().year
    pring_pos = pring_decennial_position(current_year)
    pring_info = pring_decennial_seasonality(current_year)

    if imfs is not None and imfs.size > 0:
        K = imfs.shape[0]
        window = 12
        imf_names = [f"IMF{i+1}" for i in range(K)]
        imf_slopes = {}
        for k in range(K):
            recent = imfs[k, -window:]
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            imf_slopes[imf_names[k]] = slope

        cycle_score = compute_cycle_score(imf_slopes, imf_weights)

        multi_phases = {}
        for name, slope in imf_slopes.items():
            if slope > 0:
                multi_phases[name] = "up"
            elif slope < 0:
                multi_phases[name] = "down"
            else:
                multi_phases[name] = "transition"

        weighted_up = sum(
            imf_weights.get(name, 0.25) for name, phase in multi_phases.items() if phase == "up"
        )
        weighted_total = sum(imf_weights.get(name, 0.25) for name in multi_phases)
        weighted_ratio = weighted_up / weighted_total if weighted_total > 0 else 0.5

        if weighted_ratio >= 0.75:
            composite_phase = "STRONG_BULL"
        elif weighted_ratio >= 0.50:
            composite_phase = "MIXED_RECOVERY"
        elif weighted_ratio >= 0.25:
            composite_phase = "TRANSITION"
        else:
            composite_phase = "STRONG_BEAR"
    else:
        cycle_score = 20.0
        multi_phases = {}
        composite_phase = "TRANSITION"

    coupling_score = compute_coupling_score(locked_pairs, bic_max)

    vix_latest = float(vix_series.iloc[-1]) if len(vix_series) > 0 else np.nan
    vix_score = compute_vix_score_cn(vix_latest)

    total_score = cycle_score + coupling_score + vix_score

    if total_score >= 50:
        v9_signal = 1
        signal_label = "做多 (仓位 100%)"
    elif total_score <= 30:
        v9_signal = 0
        signal_label = "做空 (仓位 0%)"
    else:
        v9_signal = -1
        signal_label = "中性 (保持上一状态)"

    recommended_allocation = get_recommended_allocation(merrill_phase)

    return CycleState(
        merrill_phase=merrill_phase,
        merrill_phase_cn=merrill_phase_cn,
        merrill_phase_num=merrill_phase_num,
        merrill_confidence=merrill_confidence,
        growth_zscore=g_zscore,
        inflation_zscore=c_zscore,
        pring_year=current_year,
        pring_position=pring_info["position"],
        pring_seasonality=pring_info["seasonality_en"],
        pring_seasonality_cn=pring_info["seasonality_cn"],
        multi_cycle_phases=multi_phases,
        composite_phase=composite_phase,
        cycle_score=cycle_score,
        coupling_score=coupling_score,
        vix_score=vix_score,
        total_score=total_score,
        v9_signal=v9_signal,
        signal_label=signal_label,
        recommended_allocation=recommended_allocation,
        gdp_proxy_name="宏观增长因子",
        cpi_proxy_name="宏观通胀因子_生活端",
        vix_value=vix_latest,
        report_date=pd.Timestamp.now(),
        data_through=data_through or latest.name,
    )
