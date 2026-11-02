# coding=utf-8
"""退出机制管理子模块.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"退出机制".

四层防线:
    1. 技术性止损 (chandelier.py)
    2. 逻辑性证伪 (logical_exit.py)
    3. 时间性强制退出 (time_exit.py)
    4. 系统性全局风控 (risk_manager.py)

编排 (exit_manager.py):
    - ExitLayers: 四层触发状态聚合
    - combine_exit_signals: 并联/串联合并
    - evaluate_exit: 一站式逐日判定 (回测引擎直接调用)
    - exit_reason: 退出原因归因

并联: 任一层触发 → 退出 (默认, 风控优先).
串联: 全部触发 → 退出 (信号最强).
"""
from __future__ import annotations

from .chandelier import (
    chandelier_exit_signal,
    chandelier_stop_prices,
    make_chandelier_factor,
)
from .exit_manager import (
    ExitLayers,
    combine_exit_signals,
    evaluate_exit,
    exit_reason,
)
from .logical_exit import (
    logical_exit_check,
    logical_exit_for_trend,
    regime_change_signals,
)
from .risk_manager import (
    DrawdownCircuitBreaker,
    rolling_drawdown,
    stress_test_extreme_days,
    target_vol_leverage,
)
from .range_exit import (
    range_rsi_exit_check,
    range_rsi_exit_signal,
)
from .time_exit import (
    holding_days,
    time_exit_check,
    time_exit_signals,
)

__all__ = [
    # chandelier
    "chandelier_stop_prices",
    "chandelier_exit_signal",
    "make_chandelier_factor",
    # logical
    "regime_change_signals",
    "logical_exit_for_trend",
    "logical_exit_check",
    # time
    "holding_days",
    "time_exit_signals",
    "time_exit_check",
    # risk
    "DrawdownCircuitBreaker",
    "target_vol_leverage",
    "rolling_drawdown",
    "stress_test_extreme_days",
    # range exit (研报原文: 震荡交易 RSI 回落退出)
    "range_rsi_exit_signal",
    "range_rsi_exit_check",
    # manager
    "ExitLayers",
    "combine_exit_signals",
    "evaluate_exit",
    "exit_reason",
]
