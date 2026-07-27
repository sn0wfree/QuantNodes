# coding=utf-8
"""v11 海龟数学升级版 — 5 层架构 + ACT-1/2/3.

基于 10_TURTLE_TRADING_MATHEMATICS.md:
- ACT-1: yang_zhang_vol 替换 realized_vol
- ACT-2: kelly_audit 审计
- ACT-3: drawdown_controller 回撤控制
"""
from __future__ import annotations

from .config_v11 import V10Config as V11Config
from .v11_strategy import V11Strategy, run_v11
from .backtest_v11 import run_v11_backtest, V11BacktestResult
from .risk_layer_v11 import RiskLayerV11

__all__ = [
    # 配置
    "V11Config",
    # 策略
    "V11Strategy",
    "run_v11",
    # 回测
    "run_v11_backtest",
    "V11BacktestResult",
    # 风控
    "RiskLayerV11",
]
