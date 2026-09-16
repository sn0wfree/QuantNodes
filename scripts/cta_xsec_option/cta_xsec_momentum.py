# coding=utf-8
"""Stage 0: CTA 截面动量信号 (§26 S2 改进版).

改进: 多时间尺度加权 (4/13/26/52周) + 品种池限定27品种 + G1/G2/G2b门槛.

门槛:
    G1:  OOS Sharpe > 0.15 且 IS/OOS 同向
    G2:  截面宽度 >= MIN_PRODUCTS 的交易日占比 >= 60%
    G2b: 毛价差 > MIN_GROSS (5%/年)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.cta_shared.factor_research import vol_target_backtest

from .config import (
    COST_BPS,
    END,
    IS_END,
    MIN_GROSS,
    MIN_PRODUCTS,
    MOM_WINDOWS,
    TV,
    load_smoothed_panel,
)


def compute_mom_score(close: pd.DataFrame) -> pd.DataFrame:
    """多时间尺度动量得分: 4/13/26/52 周收益等权均值.

    close 为日频面板; 周期按 5 日/周折算.
    """
    rets = close.pct_change(fill_method=None)
    parts = []
    for w in MOM_WINDOWS:
        days = w * 5
        parts.append(close.pct_change(days, fill_method=None))
    score = pd.concat(parts, axis=1, keys=MOM_WINDOWS).groupby(level=1, axis=1).mean()
    score = score.where(rets.notna().cummax().astype(bool))
    return score


def build_cta_signal(
    close: pd.DataFrame,
    tier: int = 3,
    weekly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """截面排序 → 三值信号 (±1/n_active).

    返回:
        (sig, score) — sig: 日频信号矩阵 (周锚点ffill), score: 动量得分
    """
    score = compute_mom_score(close)
    ranks = score.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    sig[ranks >= 1 - 1 / tier] = 1.0
    sig[ranks <= 1 / tier] = -1.0
    sig[score.isna()] = 0.0
    if weekly:
        anchors = score.index.to_series().resample("W-FRI").last().dropna()
        sig = sig.loc[sig.index.isin(anchors)].reindex(sig.index).ffill().fillna(0)
    n_active = (sig != 0).sum(axis=1).clip(lower=1)
    sig = sig.div(n_active, axis=0)
    return sig, score


def run_cta_xsec_bt(
    close: pd.DataFrame | None = None,
    end: str = END,
    is_end: str = IS_END,
    cost_bps: float = COST_BPS,
    tv: float = TV,
) -> dict:
    """Stage 0 回测 + G1/G2/G2b 门槛评估."""
    if close is None:
        close = load_smoothed_panel()
    close = close[close.index <= end]

    sig, score = build_cta_signal(close)
    rets = close.pct_change(fill_method=None)
    res = vol_target_backtest(
        sig, close, cost_bps=cost_bps, target_vol=tv, pit_shift=1, ret_df=rets,
    )
    r = res["nav"].pct_change(fill_method=None).fillna(0)

    n = (sig != 0).sum(axis=1).clip(lower=1)
    long_leg = (sig.clip(lower=0) * rets).sum(axis=1) / n
    short_leg = (-sig.clip(upper=0) * rets).sum(axis=1) / n
    gross_ann = (long_leg - short_leg).mean() * 252

    width = score.notna().sum(axis=1)
    width_ok = (width >= MIN_PRODUCTS).mean()

    def s(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 50 or x.std() == 0:
            return float("nan")
        return x.mean() / x.std() * np.sqrt(252)

    nav = (1 + r).cumprod()
    full = s(r)
    is_s = s(r[r.index < is_end])
    oos_s = s(r[r.index >= is_end])

    same_dir = (is_s > 0) == (oos_s > 0) if not (pd.isna(is_s) or pd.isna(oos_s)) else False
    g1 = (not pd.isna(oos_s)) and oos_s > 0.15 and same_dir
    g2 = width_ok >= 0.60
    g2b = gross_ann > MIN_GROSS

    return {
        "sharpe": full,
        "ann_ret": nav.iloc[-1] ** (252 / len(r)) - 1,
        "max_dd": (nav / nav.cummax() - 1).min(),
        "is_sharpe": is_s,
        "oos_sharpe": oos_s,
        "gross_ann": gross_ann,
        "width_ok": width_ok,
        "turnover": res["turnover_annual"],
        "g1": bool(g1),
        "g2": bool(g2),
        "g2b": bool(g2b),
        "nav": nav,
        "ret": r,
        "sig": sig,
        "score": score,
    }
