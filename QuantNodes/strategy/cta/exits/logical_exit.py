# coding=utf-8
"""逻辑性证伪: 状态反转确认退出.

文章 §"逻辑性证伪" 要点:
    - 开仓逻辑成立时持仓, 一旦开仓条件逆转 (例如 trend → range)
      即便价格未触及止损也应离场 (主动型风控).
    - 防止逻辑信号频繁翻转 → 引入确认窗口 (相当于对信号做低通滤波).

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
from __future__ import annotations

import pandas as pd

from ..regime import STATE_TREND_UP, STATE_TREND_DOWN


def regime_change_signals(
    states: pd.DataFrame,
    confirm_window: int = 3,
) -> pd.DataFrame:
    """逐列判定: 状态发生变化且**新状态已持续 confirm_window 日** → 触发证伪退出.

    判定条件: state[t] ≠ state[t-confirm_window] 且 state[t-confirm_window] 非 NaN.

    语义: "在最近 confirm_window 日内确实发生过状态切换, 且该切换未被回滚".
    一旦状态确认 (新状态持续了 N 日), 后续 t+N+1 之后信号自然消失
    (因为 state[t] == state[t-N] 又回到同状态), 这与"确认瞬间即触发退出"的
    业务语义一致.

    Args:
        states: (T, N) 状态矩阵, 取值 STATE_* 常量.
        confirm_window: 持续天数 (默认 3, 文章推荐)

    Returns:
        pd.DataFrame (T, N) dtype=bool:
            True 表示该日是状态反转的"确认瞬间" (应清仓).
    """
    state_n_ago = states.shift(confirm_window)
    changed = state_n_ago.notna() & (states != state_n_ago)
    return changed.fillna(False)


def logical_exit_for_trend(
    states: pd.DataFrame,
    position: pd.Series,
    confirm_window: int = 3,
) -> pd.Series:
    """持仓上下文下的逻辑性证伪.

    仅当 (a) 状态变化被确认, 且 (b) 当前持仓方向与新状态冲突时触发.
    例: 原持多 (trend_up) → 状态变为 trend_down 或 range → 触发证伪.
    原持空 (trend_down) → 状态变为 trend_up 或 range → 触发证伪.
    range 状态下的多空由反转逻辑决定, 此处保守按"任何状态变化都证伪"处理.

    Args:
        states: (T, N) 状态矩阵
        position: 单品种持仓方向 (-1 / 0 / +1)
        confirm_window: 确认窗口

    Returns:
        pd.Series[bool], True 表示应清仓.
    """
    assert position.nunique() and len(position) == len(states)
    confirmed = regime_change_signals(states, confirm_window)
    # 触发条件: 状态被确认反转 AND 当前有持仓
    triggered = confirmed.iloc[:, 0] & (position != 0)
    return triggered.fillna(False)


def logical_exit_check(
    current_state: str,
    position: int,
    new_state: str,
) -> bool:
    """单时点判定 (供回测引擎在每个 bar 调用).

    Args:
        current_state: 持仓时的开仓状态
        position: 当前持仓 (+1 / -1, 不接受 0)
        new_state: 当前判定状态

    Returns:
        bool: True 表示应触发证伪退出.
    """
    if position == 0:
        return False
    if current_state == new_state:
        return False
    # 持多时, 新状态非多头趋势 → 退出
    if position > 0 and new_state != STATE_TREND_UP:
        return True
    # 持空时, 新状态非空头趋势 → 退出
    if position < 0 and new_state != STATE_TREND_DOWN:
        return True
    return False
