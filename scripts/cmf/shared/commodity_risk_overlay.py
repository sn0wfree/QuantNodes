# coding=utf-8
"""commodity_risk_overlay.py — 通用商品三级风控覆盖层.

数据源无关 + 频率无关: 接收价格/仓位序列, 通过 periods_per_year 适配日频/周频.

三级风控:
    Level 1 - 波动率压制 (continuous):
        realized_vol > vol_cap 时, 仓位×(vol_cap/realized_vol)

    Level 2 - 软减仓 (discrete):
        rolling_dd < soft_cap 时, 仓位×0.5

    Level 3 - 硬止损 + 智能再入场:
        触发: rolling_dd < hard_cap OR 单期收益 < crash_cap
        动作: 立即清仓 + 冷却期
        再入场: 冷却期结束后需 confirm_low 低分位 AND confirm_z 低z值
                (confirm序列可选, 无则冷却期后直接恢复)

参考实现:
    - QuantNodes/strategy/commodity_fundamental/gold/gold_trend_bnh.py (compute_risk_overlay)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_drawdown(close: pd.Series, position: pd.Series, window: int = 252) -> pd.Series:
    """滚动窗口最大回撤 (仅在持仓期间).

    Args:
        close: 价格序列
        position: 仓位序列
        window: 滚动窗口

    Returns:
        滚动回撤序列 (负值, 空仓期为0)
    """
    ret = close.pct_change()
    strat_ret = (position.shift(1) * ret).fillna(0)
    dd = pd.Series(np.nan, index=close.index)
    for i in range(window, len(close)):
        w = strat_ret.iloc[i - window:i + 1]
        if (position.iloc[i - window:i + 1] > 0).any():
            nav = (1 + w).cumprod()
            dd.iloc[i] = (nav / nav.cummax() - 1).min()
        else:
            dd.iloc[i] = 0.0
    return dd


def compute_risk_overlay(
    close: pd.Series,
    position: pd.Series,
    hard_cap: float = -0.15,
    soft_cap: float = -0.12,
    crash_cap: float | None = -0.03,
    vol_cap: float = 0.20,
    cooldown_periods: int = 5,
    vol_window: int = 20,
    dd_window: int = 252,
    periods_per_year: int = 252,
    confirm_low: pd.Series | None = None,
    confirm_z: pd.Series | None = None,
    reentry_low_pct: float = 0.6,
    reentry_z: float = 1.0,
    z_window: int = 90,
    reentry_ok: pd.Series | None = None,
    warmup_periods: int = 60,
) -> pd.Series:
    """通用三级风控覆盖层.

    Args:
        close: 价格序列 (日频或周频)
        position: 原始仓位 (0~上限)
        hard_cap: 滚动回撤硬阈值 (触发清仓), None则不启用
        soft_cap: 滚动回撤软阈值 (仓位×0.5)
        crash_cap: 单期收益暴跌阈值 (触发清仓), None则不启用
        vol_cap: 年化波动率上限 (连续压制)
        cooldown_periods: 冷却期长度 (周期数)
        vol_window: 波动率窗口
        dd_window: 回撤窗口
        periods_per_year: 年化周期数 (日频252, 周频52)
        confirm_low: 再入场确认序列 (低分位利好, 如VIX)
        confirm_z: 再入场确认序列 (低z值利好, 如GVZ)
        reentry_low_pct: confirm_low 分位数阈值
        reentry_z: confirm_z z值阈值
        z_window: confirm_z 的zscore窗口
        reentry_ok: 通用再入场确认 (布尔序列, True允许恢复); 提供时优先生效,
                    忽略 confirm_low/confirm_z
        warmup_periods: 前N期不应用风控

    Returns:
        调整后仓位序列
    """
    pos = position.reindex(close.index).ffill().fillna(0).copy()

    if confirm_low is not None:
        confirm_low = confirm_low.reindex(close.index).ffill()
    if confirm_z is not None:
        confirm_z = confirm_z.reindex(close.index).ffill()

    daily_ret = close.pct_change()
    rolling_dd = rolling_drawdown(close, pos, window=dd_window)

    # Level 1: 波动率压制
    realized_vol = daily_ret.rolling(vol_window, min_periods=5).std() * np.sqrt(periods_per_year)
    vol_mask = (realized_vol > vol_cap) & (realized_vol.notna())
    vol_scale = (vol_cap / realized_vol).clip(upper=1.0)
    pos = pos.where(~vol_mask, pos * vol_scale)

    # Level 2: 软减仓
    soft_mask = rolling_dd < soft_cap
    pos = pos.where(~soft_mask, pos * 0.5)

    # Level 3: 硬止损 (清零)
    hard_mask = pd.Series(False, index=close.index)
    if hard_cap is not None:
        hard_mask = rolling_dd < hard_cap
    if crash_cap is not None:
        hard_mask = hard_mask | (daily_ret < crash_cap)
    hard_mask = hard_mask.fillna(False)
    pos = pos.where(~hard_mask, 0.0)

    # 智能再入场
    final_pos = pos.copy()
    cooldown_until = pd.Series(pd.NaT, index=close.index)

    if cooldown_periods > 0:
        trigger = hard_mask & (~hard_mask.shift(1).fillna(False))
        for i in range(len(close)):
            if trigger.iloc[i]:
                end_i = min(i + cooldown_periods, len(close) - 1)
                cooldown_until.iloc[i:end_i + 1] = close.index[end_i]

    if reentry_ok is not None:
        reentry_ok = reentry_ok.reindex(close.index).fillna(False).astype(bool)
        for i in range(len(close)):
            if pd.notna(cooldown_until.iloc[i]) and close.index[i] <= cooldown_until.iloc[i]:
                if close.index[i] == cooldown_until.iloc[i]:
                    if not reentry_ok.iloc[i]:
                        new_end = min(i + cooldown_periods, len(close) - 1)
                        cooldown_until.iloc[i:new_end + 1] = close.index[new_end]
                        final_pos.iloc[i] = 0.0
                else:
                    final_pos.iloc[i] = 0.0
    elif confirm_low is not None and confirm_z is not None:
        z_mean = confirm_z.rolling(z_window, min_periods=20).mean()
        z_std = confirm_z.rolling(z_window, min_periods=20).std().replace(0, np.nan)
        z_score = (confirm_z - z_mean) / z_std
        low_pct = confirm_low.rolling(dd_window, min_periods=20).rank(pct=True)

        for i in range(len(close)):
            if pd.notna(cooldown_until.iloc[i]) and close.index[i] <= cooldown_until.iloc[i]:
                if close.index[i] == cooldown_until.iloc[i]:
                    v_pct = low_pct.iloc[i]
                    g_z = z_score.iloc[i]
                    if pd.notna(v_pct) and pd.notna(g_z):
                        if v_pct < reentry_low_pct and g_z < reentry_z:
                            pass  # 允许恢复
                        else:
                            new_end = min(i + cooldown_periods, len(close) - 1)
                            cooldown_until.iloc[i:new_end + 1] = close.index[new_end]
                            final_pos.iloc[i] = 0.0
                else:
                    final_pos.iloc[i] = 0.0
    elif cooldown_periods > 0:
        # 无确认序列: 冷却期内强制空仓, 结束后直接恢复
        for i in range(len(close)):
            if pd.notna(cooldown_until.iloc[i]) and close.index[i] < cooldown_until.iloc[i]:
                final_pos.iloc[i] = 0.0

    final_pos = final_pos.where(final_pos.notna(), pos)

    # warmup期: 不应用风控
    final_pos.iloc[:warmup_periods] = pos.iloc[:warmup_periods]

    return final_pos


__all__ = [
    "compute_risk_overlay",
    "rolling_drawdown",
]
