# coding=utf-8
"""v6 策略模块 — 合并 v6 + v6_1 + v6_2.

v6:   v1.0 风控框架 + v5 选股 + v5.1 逆波动加权
v6_1: IC-IR 加权 (expanding window, 自动剔除失效因子)
v6_2: 因子正交化 + IC 加权 (已 DEPRECATED, 被 v7.10 超越)
"""
from .industry_rotation_v6 import V6SubStrategy, V6Config, run_v6_backtest
from .industry_rotation_v6_1 import V6_1Config, V6_1SubStrategy, select_v6_1, run_v6_1_backtest
from .industry_rotation_v6_2 import V6_2Config, V6_2SubStrategy, run_v6_2_backtest
from .factor_weighting import (
    compute_cross_section_ic,
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
)
from .factor_orthogonal import get_factor_ir_order, orthogonalize_factor_panel

__all__ = [
    # v6 (等权 + 逆波动)
    "V6SubStrategy", "V6Config", "run_v6_backtest",
    # v6_1 (IC-IR 加权)
    "V6_1Config", "V6_1SubStrategy", "select_v6_1", "run_v6_1_backtest",
    # v6_2 (正交化, DEPRECATED)
    "V6_2Config", "V6_2SubStrategy", "run_v6_2_backtest",
    # 公共函数
    "compute_cross_section_ic", "compute_ic_timeseries",
    "compute_factor_weights", "align_weights_with_rebal_dates",
    "get_factor_ir_order", "orthogonalize_factor_panel",
]
