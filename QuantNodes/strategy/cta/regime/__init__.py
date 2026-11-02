# coding=utf-8
"""Regime 状态识别子模块.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).

公开 API:
    估计:
        - ARGarchEstimator: (μ, ρ, σ) 估计, 支持 Kaufman 自适应窗口
                            与可选 GARCH(1,1) 增强
        - kaufman_adaptive_window: Kaufman 效率比 → 自适应窗口长度

    TSI Bootstrap:
        - tsi_point: 单点 TSI = μ · √T / σ
        - tsi_bootstrap_ci: TSI 解析置信区间 (参数化近似)
        - compute_tsi_panel: 批量 TSI 面板

    状态分类:
        - classify_regime: 趋势/震荡 规则分类
        - classify_regime_from_estimator: 一站式 (estimator + tsi → states)
        - 常量: STATE_UNKNOWN / STATE_RANGE / STATE_TREND_UP / STATE_TREND_DOWN
"""
from __future__ import annotations

from .ar_garch_estimator import (
    ARGarchEstimator,
    kaufman_adaptive_window,
)
from .regime_classifier import (
    STATE_RANGE,
    STATE_TREND_DOWN,
    STATE_TREND_UP,
    STATE_UNKNOWN,
    classify_regime,
    classify_regime_from_estimator,
)
from .tsi_bootstrap import (
    compute_tsi_panel,
    tsi_bootstrap_ci,
    tsi_point,
)

__all__ = [
    # estimator
    "ARGarchEstimator",
    "kaufman_adaptive_window",
    # tsi bootstrap
    "tsi_point",
    "tsi_bootstrap_ci",
    "compute_tsi_panel",
    # classifier
    "STATE_UNKNOWN",
    "STATE_RANGE",
    "STATE_TREND_UP",
    "STATE_TREND_DOWN",
    "classify_regime",
    "classify_regime_from_estimator",
]
