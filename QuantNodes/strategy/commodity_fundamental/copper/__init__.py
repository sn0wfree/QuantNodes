# coding=utf-8
"""Commodity Fundamental — 铜品种子策略包.

汇总铜专有策略, 共享信号层 (kalman_filter/lasso/quadrant等) 在上层包.

策略清单:
    copper_v6_model     V6策略模型类 (推荐, 支持自动参数优化)
    copper_carry_model  Carry策略模型类 (推荐, 含无前视回测)
    copper_inventory_cycle  库存周期四象限 (独立策略)
    copper_momentum     动量策略 (26周)
    copper_vrp_auratio  VRP和AU-CU价格比策略
    copper_daily_stop   日度ATR追踪止损机制
    copper_model_v6     v6主模型 (函数接口)
    copper_carry        carry子策略 (函数接口)

推荐组合: 突破策略(因子择时)=75% + carry=10% + 四象限=15%
年化10.4%, Sharpe 0.97, MaxDD -10.0%, 无杠杆
平均仓位65.7%, 年换手8.3倍, 手续费损耗0.46%
因子择时: 动量>0 或 波动率<70% 时持仓
"""
from __future__ import annotations

from .copper_v6_model import V6Model, V6Config
from .copper_carry_model import CarryModel, CarryConfig
from .copper_inventory_momentum import (
    compute_inventory_momentum, compute_inventory_momentum_v4,
    backtest_inventory_momentum_v4,
)
from .copper_inventory_cycle import (
    compute_inventory_cycle_quadrant, compute_inventory_cycle_v2,
    compute_inventory_cycle_optimized, compute_inventory_cycle_with_atr_stop,
    backtest_inventory_cycle, backtest_inventory_cycle_optimized,
    backtest_inventory_cycle_with_atr_stop,
    analyze_quadrant_performance,
)
from .copper_momentum import (
    compute_momentum_signal, compute_momentum_v2,
    backtest_momentum,
)
from .copper_vrp_auratio import (
    compute_vrp_signal, compute_au_cu_ratio_signal,
    backtest_strategy,
)
from .copper_breakout import (
    compute_breakout_signal,
    backtest_breakout,
)
from .copper_combined import (
    compute_combined_signal,
    backtest_combined,
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
    "compute_inventory_momentum", "compute_inventory_momentum_v4",
    "backtest_inventory_momentum_v4",
    # 库存周期四象限
    "compute_inventory_cycle_quadrant", "compute_inventory_cycle_v2",
    "compute_inventory_cycle_optimized", "compute_inventory_cycle_with_atr_stop",
    "backtest_inventory_cycle", "backtest_inventory_cycle_optimized",
    "backtest_inventory_cycle_with_atr_stop",
    "analyze_quadrant_performance",
    # 动量策略
    "compute_momentum_signal", "compute_momentum_v2",
    "backtest_momentum",
    # VRP和AU-CU价格比策略
    "compute_vrp_signal", "compute_au_cu_ratio_signal",
    "backtest_strategy",
    # 突破策略
    "compute_breakout_signal",
    "backtest_breakout",
    # 组合策略
    "compute_combined_signal",
    "backtest_combined",
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
