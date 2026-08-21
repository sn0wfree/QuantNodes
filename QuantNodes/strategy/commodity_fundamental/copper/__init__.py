# coding=utf-8
"""Commodity Fundamental — 铜品种子策略包.

汇总铜专有策略, 共享信号层 (kalman_filter/lasso/quadrant等) 在上层包.

策略清单:
    copper_model_v6    v6主模型(当前): avg(v4仓位, v5精简仓位) + bull_enhance + 波动率目标化
    copper_carry       期限结构/Carry子策略 (long-only, 组合层独立配置)
    copper_model_v5    v5: 波动率+动量+经济状态 (use_econ=False供v6调用)
    copper_model_v4    v4: VIX+DXY+库存象限 (v6组件)
    copper_model_v6_3  v6.3多因子体系(实验): 12因子等权 + bull_enhance
    copper_pricing     铜定价模型 (无信号入口)
    copper_economy     美债期限差经济状态信号 (消融证实负贡献, 已不进主栈)
    copper_ml          ML实验版 (已归档: 0.62 < v4 0.71, 同族高相关)

当前策略来源: v6(主) + carry(避险) 组合层并联
"""
from __future__ import annotations

from .copper_model_v6 import (
    build_signal_v6, backtest_v6, apply_bull_enhance,
)
from .copper_model_v5 import build_signal_v5, backtest_v5
from .copper_model_v4 import build_signal_v4, backtest_v4
from .copper_carry import (
    build_signal_carry,
    backtest_carry,
    load_term_structure,
    compute_carry_signal,
)
from .copper_model_v6_3 import build_signal_v6_3, backtest_v6_3
from .copper_factors import compute_all_factors
from .copper_factor_groups import combine_all_groups

__all__ = [
    "build_signal_v6",
    "backtest_v6",
    "apply_bull_enhance",
    "build_signal_v5",
    "backtest_v5",
    "build_signal_v4",
    "backtest_v4",
    "build_signal_carry",
    "backtest_carry",
    "load_term_structure",
    "compute_carry_signal",
    "build_signal_v6_3",
    "backtest_v6_3",
    "compute_all_factors",
    "combine_all_groups",
]
