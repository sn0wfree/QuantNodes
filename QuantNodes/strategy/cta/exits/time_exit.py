# coding=utf-8
"""时间性强制退出: 预期获利窗口.

文章 §"时间性强制退出":
    - 价格未触及止损, 逻辑未逆转, 但预期趋势迟迟未兑现.
    - 继续持仓 = 锁定资金, 错过更优机会 → 机会成本.
    - 组合层面强制释放资金: 预设预期获利窗口, 若未达标则强制平仓.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
from __future__ import annotations

import pandas as pd


def holding_days(position: pd.Series) -> pd.Series:
    """当前已持仓天数 (从最近一次 0 开始累计).

    Args:
        position: 持仓方向 Series, 取值 {-1, 0, +1}; 0 表示空仓.

    Returns:
        pd.Series[int], 每个时点已持仓天数 (空仓时为 0).
    """
    is_open = (position != 0).astype(int)
    # 累计非零段: 用 (cumsum - cumsum_at_last_zero) 思路
    grp = is_open.groupby((is_open != is_open.shift()).cumsum()).cumsum()
    return grp.where(is_open == 1, 0).astype("Int64")


def time_exit_signals(
    position: pd.Series,
    unrealized_pnl: pd.Series,
    max_holding_days: int,
    min_pnl_threshold: float = 0.0,
) -> pd.Series:
    """时间性强制退出信号.

    触发条件 (同时满足):
        1. 当前持仓 > 0 (有仓位)
        2. 持仓天数 ≥ max_holding_days
        3. 浮盈 < min_pnl_threshold (尚未积累足够利润)

    Args:
        position: 持仓方向 Series (-1/0/+1)
        unrealized_pnl: 当前持仓的浮盈浮亏 (与 position 同索引)
        max_holding_days: 最长持仓天数
        min_pnl_threshold: 最低盈利要求 (默认 0, 即只要未盈利)

    Returns:
        pd.Series[bool], True 表示应强制退出.
    """
    days = holding_days(position)
    triggered = (
        (days >= max_holding_days)
        & (position != 0)
        & (unrealized_pnl < min_pnl_threshold)
    )
    return triggered.fillna(False)


def time_exit_check(
    position: int,
    days_held: int,
    unrealized_pnl: float,
    max_holding_days: int,
    min_pnl_threshold: float = 0.0,
) -> bool:
    """单时点判定 (供回测引擎调用)."""
    if position == 0:
        return False
    if days_held < max_holding_days:
        return False
    return unrealized_pnl < min_pnl_threshold
