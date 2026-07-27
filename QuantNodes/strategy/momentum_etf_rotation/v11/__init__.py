# coding=utf-8
"""v11 海龟数学升级版 — 5 层架构 + ACT-1/2/3.

历史:
  - 5 层架构原本在 QuantNodes/.../v10/ (Stage 33 计划)
  - 2026-07-27 正式迁移到 v11
  - v10/ 现只保留 4 策略主体 (4 策略 Vol-parity, 生产版)

基于 docs/research_history/10_TURTLE_TRADING_MATHEMATICS.md:
- ACT-1: yang_zhang_vol 替换 realized_vol
- ACT-2: kelly_audit 审计
- ACT-3: drawdown_controller 回撤控制
"""
from __future__ import annotations

# 配置 (全部 Config 类族)
from .config_v11 import V10Config as V11Config
from .config_v11 import (
    MacroLayerConfig,
    IndustryLayerConfig,
    StyleLayerConfig,
    FactorLayerConfig,
    RiskLayerConfig,
    PositionLayerConfig,
    PortfolioLayerConfig,
)

# 5 层模块
from .macro_layer import MacroLayer, compute_macro_signal
from .industry_layer import IndustryLayer, compute_industry_tilt
from .style_layer import StyleLayer, compute_style_weights
from .factor_layer import FactorLayer, compute_factor_tilt
from .risk_layer import RiskLayer, compute_bear_probability
from .position_layer import PositionLayer, compute_dynamic_position
from .portfolio_layer import PortfolioLayer, build_final_weights

# ACT-2/3 风控 (v11 新增)
from .risk_layer_v11 import RiskLayerV11

# 主入口
from .v11_strategy import V11Strategy, run_v11

# 回测
from .backtest_v11 import run_v11_backtest, V11BacktestResult

__all__ = [
    # 配置
    "V11Config",
    "MacroLayerConfig", "IndustryLayerConfig", "StyleLayerConfig",
    "FactorLayerConfig", "RiskLayerConfig", "PositionLayerConfig",
    "PortfolioLayerConfig",
    # 5 层模块
    "MacroLayer", "IndustryLayer", "StyleLayer", "FactorLayer",
    "RiskLayer", "PositionLayer", "PortfolioLayer",
    "compute_macro_signal", "compute_industry_tilt",
    "compute_style_weights", "compute_factor_tilt",
    "compute_bear_probability", "compute_dynamic_position",
    "build_final_weights",
    # ACT-2/3
    "RiskLayerV11",
    # 主入口
    "V11Strategy", "run_v11",
    # 回测
    "run_v11_backtest", "V11BacktestResult",
]
