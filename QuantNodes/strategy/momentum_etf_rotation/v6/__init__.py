# coding=utf-8
"""v6 策略模块 (Stage 26) — v1.0 风控框架 + v5 选股 + v5.1 逆波动加权.

[注] v6 没有自己的 select (选股继承自 v5), 仅用 v5.1 的 weight (逆波动).

设计:
- 选股层 (v5): 11 量价因子 + 截面 z-score + Top-N
- 加权层 (v5.1): 逆波动率 (60日窗口, vol_floor=0.01, T+1 lag) + max_weight=0.25
- 风控层 (v2 框架): VT (波动率目标) + TF (趋势过滤) + Cost (交易成本)
  - 完全复用 v2.RotationConfig, v2.apply_vol_targeting_v2,
    v2.apply_trend_filter_v2, v2.calculate_turnover_cost_v2

回测目标 (口径 A 含 5bp 成本, 2018-2026):
- OOS Calmar: v1.0 (1.791) + v5.1 (0.604) → 目标 0.8-1.0
- 全期 Calmar: ≥ v5.1 baseline (0.821)
- OOS DD: ≤ v5.1 (18.04%) → 期望 ≤ -10%

参考:
- v1.0 框架: v2/strategy_versions_v2.py:v1_0_v2
- v5.1 选股 + 加权: v5_1/industry_rotation_v5_1.py
- 风控层 API: v2/portfolio_v2.py
"""
from .industry_rotation_v6 import (
    V6SubStrategy,
    V6Config,
    run_v6_backtest,
)

__all__ = [
    "V6SubStrategy",
    "V6Config",
    "run_v6_backtest",
]
