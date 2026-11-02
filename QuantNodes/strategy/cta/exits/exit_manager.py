# coding=utf-8
"""退出机制编排 (并联 / 串联).

文章 §"退出机制" 收尾:
    每一种退出机制各有优劣, 有的偏左侧 (技术止损), 有的偏右侧 (时间强制);
    实战中往往需要将多种退出机制进行并联或串联, 建立系统性风控体系.

    - parallel (并联): 任一层触发 → 退出 (默认, 偏左侧, 风控优先)
    - serial   (串联): 所有层全部触发 → 退出 (偏右侧, 信号最强才动)

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .chandelier import chandelier_exit_signal
from .logical_exit import logical_exit_for_trend
from .time_exit import time_exit_signals


@dataclass
class ExitLayers:
    """退出机制四层防线的当前状态 (按品种逐日).

    每个字段为 pd.Series[bool] (True=该层触发退出).
    """

    technical: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    logical: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    time_based: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    risk_halt: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))

    def any(self) -> pd.Series:
        return self.technical | self.logical | self.time_based | self.risk_halt

    def all(self) -> pd.Series:
        return self.technical & self.logical & self.time_based & self.risk_halt


def combine_exit_signals(
    layers: ExitLayers,
    mode: Literal["parallel", "serial"] = "parallel",
) -> pd.Series:
    """组合四层退出信号.

    Args:
        layers: 四层退出机制的触发状态
        mode: 'parallel' (任一触发) 或 'serial' (全部触发)

    Returns:
        pd.Series[bool]: 合并后的最终退出信号.
    """
    if mode == "parallel":
        return layers.any()
    if mode == "serial":
        return layers.all()
    raise ValueError(f"mode must be 'parallel' or 'serial', got {mode!r}")


def evaluate_exit(
    position: pd.Series,
    ohlc: pd.DataFrame,
    states: pd.DataFrame | None = None,
    unrealized_pnl: pd.Series | None = None,
    max_holding_days: int = 60,
    chandelier_kwargs: dict | None = None,
    logical_confirm_window: int = 3,
    risk_halt: pd.Series | None = None,
    mode: Literal["parallel", "serial"] = "parallel",
) -> pd.Series:
    """一站式退出判定 (向量化, 供回测引擎每日调用).

    Args:
        position: 持仓方向 (-1/0/+1)
        ohlc: OHLC 面板 (含 high/low/close, 需与 position 同索引)
        states: 状态矩阵 (可选, 若 None 则跳过逻辑性证伪层)
        unrealized_pnl: 浮盈浮亏 (可选, 若 None 则跳过时间性强制层)
        max_holding_days: 最长持仓天数
        chandelier_kwargs: 透传给 chandelier_exit_signal 的参数
        logical_confirm_window: 状态确认窗口
        risk_halt: 全局熔断信号 (可选, 全仓共享)
        mode: parallel / serial

    Returns:
        pd.Series[bool]: True 表示该日应清仓.
    """
    chandelier_kwargs = chandelier_kwargs or {}
    tech = chandelier_exit_signal(ohlc, position, **chandelier_kwargs)
    logical = (
        logical_exit_for_trend(states, position, logical_confirm_window)
        if states is not None
        else pd.Series(False, index=position.index)
    )
    time_b = (
        time_exit_signals(position, unrealized_pnl, max_holding_days)
        if unrealized_pnl is not None
        else pd.Series(False, index=position.index)
    )
    risk = risk_halt if risk_halt is not None else pd.Series(False, index=position.index)

    layers = ExitLayers(technical=tech, logical=logical, time_based=time_b, risk_halt=risk)
    return combine_exit_signals(layers, mode=mode)


def exit_reason(
    layers: ExitLayers,
) -> pd.Series:
    """逐日返回退出原因 (字符串), 便于回测归因.

    取第一个触发的层 (按 technical → logical → time_based → risk_halt 顺序).
    若多层同时触发, 只报告最先匹配者.
    """
    none = np.array([""] * len(layers.technical), dtype=object)
    out = pd.Series(none, index=layers.technical.index)

    def _overlay(base: pd.Series, mask: pd.Series, tag: str) -> pd.Series:
        return base.mask(mask & (base == ""), tag)

    out = _overlay(out, layers.technical, "technical")
    out = _overlay(out, layers.logical, "logical")
    out = _overlay(out, layers.time_based, "time")
    out = _overlay(out, layers.risk_halt, "risk_halt")
    return out
