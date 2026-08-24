# coding=utf-8
"""Commodity Fundamental — 铜品种子策略包.

汇总铜专有策略, 共享信号层 (kalman_filter/lasso/quadrant等) 在上层包.

策略清单:
    copper_v6_model     V6策略模型类 (推荐使用, 支持自动参数优化)
    copper_carry_model  Carry策略模型类 (推荐使用)
    copper_model_v6     v6主模型 (函数接口)
    copper_carry        carry子策略 (函数接口)
    copper_model_v6_3   v6.3多因子体系 (实验)

当前策略来源: v6(主) + carry(避险) 组合层并联
"""
from __future__ import annotations

from .copper_v6_model import V6Model, V6Config
from .copper_carry_model import CarryModel, CarryConfig
from .copper_model_v6 import (
    build_signal_v6, backtest_v6, apply_bull_enhance,
)
from .copper_carry import (
    build_signal_carry,
    backtest_carry,
    load_term_structure,
    compute_carry_signal,
)
from .copper_model_v6_3 import build_signal_v6_3, backtest_v6_3

__all__ = [
    # V6模型类 (推荐)
    "V6Model",
    "V6Config",
    # Carry模型类 (推荐)
    "CarryModel",
    "CarryConfig",
    # v6函数接口
    "build_signal_v6",
    "backtest_v6",
    "apply_bull_enhance",
    # carry函数接口
    "build_signal_carry",
    "backtest_carry",
    "load_term_structure",
    "compute_carry_signal",
    # v6.3实验
    "build_signal_v6_3",
    "backtest_v6_3",
]
