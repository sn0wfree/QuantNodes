# coding=utf-8
"""shared/ — 商品CTA共享工具层.

提取自 gold/ 和 copper/ 的共用代码, 数据源无关 (价格序列作为参数传入).

模块:
    - commodity_breakout: 突破 + 追踪止损
    - commodity_momentum: 动量 + 波动率目标化
    - commodity_vol_target: 波动率目标化
    - commodity_timing: 因子开仓时机过滤
    - commodity_risk_overlay: 通用三级风控覆盖层
    - weekly_backtest: 通用周频回测引擎

使用示例:
    from scripts.cmf.shared import compute_breakout_signal, backtest_weekly

    close = load_gold_price()
    signal = compute_breakout_signal(close, lookback=20, threshold=0.95)
    result = backtest_weekly(signal["close"], signal["position"])

参考实现:
    - QuantNodes/strategy/commodity_fundamental/gold/
    - QuantNodes/strategy/commodity_fundamental/copper/
"""
from __future__ import annotations

from .commodity_breakout import compute_breakout_signal, backtest_breakout
from .commodity_momentum import compute_momentum_signal
from .commodity_vol_target import vol_target_positions, realized_volatility
from .commodity_timing import momentum_or_vol_filter, momentum_filter
from .commodity_risk_overlay import compute_risk_overlay, rolling_drawdown
from .weekly_backtest import backtest_weekly, get_weekly_returns

__all__ = [
    "compute_breakout_signal",
    "backtest_breakout",
    "compute_momentum_signal",
    "vol_target_positions",
    "realized_volatility",
    "momentum_or_vol_filter",
    "momentum_filter",
    "compute_risk_overlay",
    "rolling_drawdown",
    "backtest_weekly",
    "get_weekly_returns",
]
