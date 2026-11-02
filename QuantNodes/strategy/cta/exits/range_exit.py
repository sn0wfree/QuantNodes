# coding=utf-8
"""震荡交易的逻辑性止损 (RSI 回落退出).

研报原文 (华泰金工《时序CTA方法论综述》§"实证分析 — 离场信号"):
    对于震荡状态下的交易, 离场信号采取逻辑性止损, 这是因为逻辑性止损
    一般具有"不积跬步无以至千里"的特点, 与反转交易高胜率、低赔率的
    特点相一致。具体来说:
        - 对于 RSI 大于 70 时开的空单, 当 RSI 回落至 60 以内时平空
        - 对于 RSI 小于 30 时开的多单, 当 RSI 回升至 40 以上时平多

设计要点:
    - 退出由**开仓时的状态**决定 (开仓于 range → RSI 回落退出)
    - 趋势交易的 ATR 移动止损不适用于震荡交易:
      吊灯价位锚定 22 日极值, 在震荡区间内离入场价过近,
      会导致反转交易开仓即被"止损" (实测 98% 交易持仓仅 1 天).
"""
from __future__ import annotations

import pandas as pd


def range_rsi_exit_signal(
    position: pd.Series,
    rsi: pd.Series,
    exit_long: float = 40.0,
    exit_short: float = 60.0,
) -> pd.Series:
    """震荡交易 RSI 回落退出信号.

    语义 (研报原文):
        多单 (开仓于 RSI 超卖): 持仓中 RSI 回升至 exit_long 以上 → 平多
        空单 (开仓于 RSI 超买): 持仓中 RSI 回落至 exit_short 以内 → 平空

    Args:
        position: 持仓方向序列 (-1/0/+1), 与 rsi 同索引.
            position[t] != 0 视为"该日持仓", 用其方向判定退出条件.
        rsi: RSI 序列 (调用方预计算, 建议 14 日).
        exit_long: 多单 RSI 回升退出阈值 (研报原文 40)
        exit_short: 空单 RSI 回落退出阈值 (研报原文 60)

    Returns:
        pd.Series[bool]: True 表示该日应平仓.
    """
    if not position.index.equals(rsi.index):
        raise ValueError("position and rsi must share the same index")

    out = pd.Series(False, index=position.index)
    long_open = position > 0
    short_open = position < 0
    # 多单: RSI 回升至阈值以上 → 反转兑现 → 平仓落袋
    out[long_open & (rsi > exit_long)] = True
    # 空单: RSI 回落至阈值以内 → 反转兑现 → 平仓落袋
    out[short_open & (rsi < exit_short)] = True
    return out


def range_rsi_exit_check(
    position: int,
    rsi_value: float,
    exit_long: float = 40.0,
    exit_short: float = 60.0,
) -> bool:
    """单时点判定 (供逐日循环调用)."""
    if position == 0:
        return False
    if position > 0:
        return rsi_value > exit_long
    return rsi_value < exit_short
