# coding=utf-8
"""v10 自上而下完备资产配置框架.

5 层架构 (基于 docs/57 用户确认版):
  Layer 1: 宏观择时 (5 因子 + 熵权 + TV-PR 可选)
  Layer 2A: 行业轮动 (regime 条件 + 相关约束)
  Layer 2B: 风格轮动 (IC 驱动, 复用 v4 factor_timing)
  Layer 2C: 因子选股 (5 因子 + K=10, 复用 v9 citic_multifactor)
  Layer 3: 风险控制 (Jump Model, 复用 v8)
  Layer 4: 动态仓位 (pos + bear_prob 双控)
  Layer 5: 组合构建 (RP × tilt × pos)

核心贡献来源:
  P0 #1: 动态仓位 (v9 银河方案, Brinson 71% alpha)
  P0 #2: Jump Model 牛熊 (v8, Sharpe 1.485)
  P1 #3: 5 风格因子横截面打分 (v9 中信多因子, +0.37)
  P1 #5: TV-PR 时变β (v7, +0.15)
  P2 #8: IC 驱动因子择时 (v4, +0.10)
"""
from __future__ import annotations

from .config_v10 import (
    V10Config,
    MacroLayerConfig,
    IndustryLayerConfig,
    StyleLayerConfig,
    FactorLayerConfig,
    RiskLayerConfig,
    PositionLayerConfig,
    PortfolioLayerConfig,
)
from .macro_layer import MacroLayer, compute_macro_signal
from .industry_layer import IndustryLayer, compute_industry_tilt
from .style_layer import StyleLayer, compute_style_weights
from .factor_layer import FactorLayer, compute_factor_tilt
from .risk_layer import RiskLayer, compute_bear_probability
from .position_layer import PositionLayer, compute_dynamic_position
from .portfolio_layer import PortfolioLayer, build_final_weights
from .v10_strategy import V10Strategy, run_v10
from .backtest_v10 import run_v10_backtest, V10BacktestResult

__all__ = [
    # 配置
    "V10Config",
    "MacroLayerConfig",
    "IndustryLayerConfig",
    "StyleLayerConfig",
    "FactorLayerConfig",
    "RiskLayerConfig",
    "PositionLayerConfig",
    "PortfolioLayerConfig",
    # 层
    "MacroLayer",
    "IndustryLayer",
    "StyleLayer",
    "FactorLayer",
    "RiskLayer",
    "PositionLayer",
    "PortfolioLayer",
    # 函数
    "compute_macro_signal",
    "compute_industry_tilt",
    "compute_style_weights",
    "compute_factor_tilt",
    "compute_bear_probability",
    "compute_dynamic_position",
    "build_final_weights",
    # 主入口
    "V10Strategy",
    "run_v10",
    # 回测
    "run_v10_backtest",
    "V10BacktestResult",
]