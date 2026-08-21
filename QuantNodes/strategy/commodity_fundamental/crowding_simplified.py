# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/crowding_simplified.py

§3.3 拥挤度一票否决 (降级版 — 仅波动率分位, 无 COT 数据).

完整版公式: crowding = 2 * (0.75*D + 0.25*U) - 1
    D = 方向性持仓 (LME基金净多头/CFTC COT等) → 缺失
    U = 1 - P(4周波动率) (波动率补数)

降级版: U 单维, 因为波动率天然能起到"波动大→潜在反转"预警.
阈值: P(4周vol) >= 0.8 → 拥挤度高 → 砍仓至 0.1
解除: 回落至 0.5 以下恢复.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_percentile(
    returns: pd.Series,
    vol_window: int = 4,
    pct_window: int = 156,
) -> pd.Series:
    """4 周波动率在 156 周窗口内的百分位."""
    vol = returns.rolling(vol_window).std() * np.sqrt(52)
    pct = vol.rolling(pct_window).rank(pct=True)
    return pct.rename(f"vol_pct_{vol_window}w")


def crowding_4w_vol(
    weekly_returns: pd.Series,
    trigger_threshold: float = 0.80,
    release_threshold: float = 0.50,
    vol_window: int = 4,
    pct_window: int = 156,
) -> pd.Series:
    """基于4周波动率百分位的简化拥挤度 (0~1, 高=可能拥挤).

    Hysteresis: 触发0.8后, 必须回落至0.5以下才解除.
    """
    pct = vol_percentile(weekly_returns, vol_window, pct_window)
    raw = pct

    signal = pd.Series(0.0, index=pct.index)
    triggered = False
    for i, v in enumerate(pct.values):
        if np.isnan(v):
            signal.iloc[i] = np.nan
            continue
        if not triggered and v >= trigger_threshold:
            triggered = True
        elif triggered and v <= release_threshold:
            triggered = False
        signal.iloc[i] = 1.0 if triggered else 0.0
    return signal


def apply_crowding_override(
    base_pos: pd.Series,
    crowding_signal: pd.Series,
    crowded_position: float = 0.1,
) -> pd.Series:
    """拥挤度一票否决: 触发 → 砍至 0.1."""
    pos = base_pos.copy()
    triggered = (crowding_signal == 1.0)
    pos[triggered] = crowded_position
    return pos


__all__ = ["vol_percentile", "crowding_4w_vol", "apply_crowding_override"]