# coding=utf-8
"""v6.1 模块 (Stage 27) — v5.1.1 量价族 + IC-IR 加权.

v6.1 = v5 选股 + IC 加权 (替代 11 等权) + v5.1.1 逆波动加权.

设计动机:
- IC 诊断显示 11 因子中仅 5-6 个 OOS IR > 0
- 用 expanding window IR 加权 → 自动剔除失效因子
- 防 look-ahead: t 期权重基于截至 t-1 的历史 IC

回测目标 (口径 A 含 5bp 成本, 2018-2026, 无风控):
- OOS Calmar: ≥ v5.1.1 baseline (0.631) → 期望 0.75+
- 全期 Calmar: ≥ v5.1.1 baseline (0.856)
- OOS DD: ≤ v5.1.1 (18.0%)

参考:
- v5 选股: industry_rotation_v5.py
- v5.1.1 加权: industry_rotation_v5_1.py
- IC 加权逻辑: factor_weighting.py (本模块)
"""
from .industry_rotation_v6_1 import (
    V6_1Config,
    V6_1SubStrategy,
    select_v6_1,
    run_v6_1_backtest,
)
from .factor_weighting import (
    compute_cross_section_ic,
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
)

__all__ = [
    "V6_1Config",
    "V6_1SubStrategy",
    "select_v6_1",
    "run_v6_1_backtest",
    "compute_cross_section_ic",
    "compute_ic_timeseries",
    "compute_factor_weights",
    "align_weights_with_rebal_dates",
]
