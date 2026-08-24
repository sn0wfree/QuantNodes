# coding=utf-8
"""Commodity Fundamental — 铜品种子策略包.

汇总铜专有策略, 共享信号层 (kalman_filter/lasso/quadrant等) 在上层包.

策略清单:
    copper_v6_model     V6策略模型类 (推荐, 支持自动参数优化)
    copper_carry_model  Carry策略模型类 (推荐, 含无前视回测)
    copper_inventory_momentum  库存动量v2 (三因子确认+ATR止损)
    copper_daily_stop   日度ATR追踪止损机制
    copper_model_v6     v6主模型 (函数接口)
    copper_carry        carry子策略 (函数接口)
    copper_model_v6_3   v6.3多因子体系 (实验)
    copper_regime_detector  市场状态识别

推荐组合: v6.2=50% + carry=30% + 库存动量v2(ATR)=20%
性能: 年化8.4%, Sharpe 1.05, MaxDD -10.7%
"""
from __future__ import annotations

from .copper_v6_model import V6Model, V6Config
from .copper_carry_model import CarryModel, CarryConfig
from .copper_inventory_momentum import (
    compute_inventory_momentum, compute_inventory_momentum_v2,
    backtest_inventory_momentum, backtest_inventory_momentum_v2,
)
from .copper_daily_stop import (
    compute_atr, apply_daily_atr_stop, backtest_daily_stop,
)
from .copper_model_v6 import (
    build_signal_v6, backtest_v6, apply_bull_enhance,
)
from .copper_model_v5 import build_signal_v5, backtest_v5
from .copper_model_v4 import build_signal_v4, backtest_v4
from .copper_carry import (
    build_signal_carry, backtest_carry,
    load_term_structure, compute_carry_signal,
)
from .copper_model_v6_3 import build_signal_v6_3, backtest_v6_3
from .copper_regime_detector import RegimeDetector, RegimeConfig

__all__ = [
    # 模型类
    "V6Model", "V6Config", "CarryModel", "CarryConfig",
    # 库存动量
    "compute_inventory_momentum", "compute_inventory_momentum_v2",
    "backtest_inventory_momentum", "backtest_inventory_momentum_v2",
    # ATR止损
    "compute_atr", "apply_daily_atr_stop", "backtest_daily_stop",
    # v6函数接口
    "build_signal_v6", "backtest_v6", "apply_bull_enhance",
    "build_signal_v5", "backtest_v5", "build_signal_v4", "backtest_v4",
    # carry函数接口
    "build_signal_carry", "backtest_carry", "load_term_structure", "compute_carry_signal",
    # v6.3实验
    "build_signal_v6_3", "backtest_v6_3",
    # 市场状态
    "RegimeDetector", "RegimeConfig",
]
