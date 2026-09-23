#!/usr/bin/env python3
# coding=utf-8
"""Carry (展期收益) 多空 sleeve (§5.48).

从合约级期限结构数据构造 carry 因子, 生成美元中性多空 sleeve 收益,
可与 D 组合按权重合成 (carry 与动量/趋势正交: 横截面秩相关 +0.09).

因子构造:
  1. signal = carry / vol_60d (风险调整展期收益)
  2. 横截面 z-score (clip ±2) → 权重 ∝ z, 美元中性 (Σw=0), gross ≤ gross_cap
  3. 月频调仓 (调仓日重置, 中间持仓不变)
  4. 收益 = Σ w(t−1)·r(t) − Σ|Δw|·rate

数据: carry 由 `scripts/extract_carry.py` 从 ClickHouse daily_quote 提取
      (合约级 Close/OpenInterest → 近月/次近月年化价差).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def month_end_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """月频调仓掩码 (每月最后交易日)."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


def build_carry_sleeve(
    panel: pd.DataFrame,
    carry: pd.DataFrame,
    vol_win: int = 60,
    gross_cap: float = 1.0,
    freq: str = "ME",
    rate: float = 0.0001,
) -> tuple[pd.Series, pd.DataFrame]:
    """构造 carry 多空 sleeve.

    Args:
        panel: 价格面板 (含期货列)
        carry: 展期收益信号 (产品 × 日期, 由 extract_carry.py 生成)
        vol_win: 风险调整波动窗口
        gross_cap: 总敞口上限
        freq: 调仓频率 ("ME" 月频 / "D" 日频)
        rate: 期货单边成本 (默认 1bp)

    Returns:
        (net_returns, weights)
    """
    ret = panel.pct_change(fill_method=None)
    vol = ret.rolling(vol_win, min_periods=20).std()
    sig = carry.reindex(panel.index).ffill() / vol.replace(0, np.nan)
    cols = [c for c in sig.columns if c in panel.columns]
    sig = sig[cols]
    rb = (month_end_mask(panel.index) if freq == "ME"
          else pd.Series(True, index=panel.index))

    w = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for i, d in enumerate(panel.index):
        if i > 0 and not rb.iloc[i]:
            w.iloc[i] = w.iloc[i - 1]
            continue
        s = sig.loc[d].dropna()
        if len(s) < 6:
            continue
        z = (s - s.mean()) / s.std() if s.std() > 0 else s * 0.0
        z = z.clip(-2, 2)
        if z.abs().sum() > 0:
            w.loc[d, z.index] = z / z.abs().sum() * gross_cap

    r = pd.Series(rate, index=cols)
    port = (w.shift(1) * ret[cols]).sum(axis=1) \
        - (w.diff().abs() * r).sum(axis=1).fillna(0.0)
    port.name = "carry_sleeve"
    return port.dropna(), w


def combine_with_d(
    nav_d: pd.Series,
    carry_ret: pd.Series,
    carry_weight: float = 0.2,
    leverage: float = 1.0,
    financing_rate: float = 0.02,
) -> pd.Series:
    """D 组合 × carry sleeve 按权重合成 (含杠杆与融资).

    combined = (1−wc)·r_D + wc·r_carry,  再 × L − (L−1)·融资利率/252
    """
    idx = nav_d.index
    r_d = nav_d.pct_change().fillna(0.0)
    cs = carry_ret.reindex(idx).fillna(0.0)
    base = (1.0 - carry_weight) * r_d + carry_weight * cs
    net = base * leverage - (leverage - 1.0) * financing_rate / ANN
    nav = (1.0 + net).cumprod()
    nav.name = "D_plus_carry"
    return nav
