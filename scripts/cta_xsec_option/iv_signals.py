# coding=utf-8
"""IV 信号层 — skew_reversal(排序) + vrp_confirm/iv_regime(乘数) + 自适应降级.

组件来源 (均为已有实证先例, 不做参数扫描):
    skew_reversal: factors/vol_surface.py SkewReversal (z>1.5 多 / z<-1.5 空)
    vrp_confirm:   options_enhanced_fut_v1.py VRP与信号同向 x1.15 / 反向 x0.85
    iv_regime:     factors/vol_surface.py IVRegime (分位>0.75 x0.7 / <0.25 x1.3)

排序融合: composite = z_mom + z_skew_rev (等权起步, lambda 固定 1.0)
降级: 无 IV 数据的品种×日 → skew 项置 0 / 乘数置 1 (退化为纯动量)
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import CTA_DB

IV_DB_TABLES = {"iv": "option_iv_daily", "vrp": "vrp_daily"}


def load_iv_vrp_panel(start: str, end: str) -> pd.DataFrame:
    """品种×日 IV/VRP 面板 (option_iv_daily + vrp_daily)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    try:
        iv = con.execute(
            f"SELECT product, trade_date, atm_iv, iv_percentile "
            f"FROM {IV_DB_TABLES['iv']} "
            f"WHERE trade_date >= '{start}' AND trade_date <= '{end}'"
        ).df()
        vrp = con.execute(
            f"SELECT product, trade_date, vrp, vrp_percentile, vrp_positive, "
            f"hv_percentile FROM {IV_DB_TABLES['vrp']} "
            f"WHERE trade_date >= '{start}' AND trade_date <= '{end}'"
        ).df()
    finally:
        con.close()
    for d in (iv, vrp):
        d["trade_date"] = pd.to_datetime(d["trade_date"])
    panel = iv.merge(vrp, on=["product", "trade_date"], how="outer")
    return panel.sort_values(["product", "trade_date"]).reset_index(drop=True)


def compute_skew_series(cache: pd.DataFrame, band: float = 0.05) -> pd.Series:
    """单品种 moneyness skew: ±band 内 put IV − call IV (>0 = 恐慌定价).

    cache: 单品种 greeks cache (需含 trade_date/strike_price/spot/option_type/iv/valid)
    """
    w = cache[cache["valid"]].copy()
    if w.empty:
        return pd.Series(dtype=float)
    w["moneyness"] = np.log(w["strike_price"] / w["spot"])
    near = w[w["moneyness"].abs() <= band]
    if near.empty:
        return pd.Series(dtype=float)
    piv = near.pivot_table(
        index="trade_date", columns="option_type", values="iv", aggfunc="mean"
    )
    if "P" not in piv.columns or "C" not in piv.columns:
        return pd.Series(dtype=float)
    skew = (piv["P"] - piv["C"]).dropna()
    skew.index = pd.to_datetime(skew.index)
    return skew.sort_index()


def skew_reversal_signal(
    skew: pd.Series, z_window: int = 60, hi: float = 1.5, lo: float = -1.5,
    mode: str = "reversal",
) -> pd.Series:
    """skew zscore 极值信号 → 三值事件信号 (+1/-1/0).

    mode="reversal" (vol_surface.py IO先例): z>hi 反转做多 / z<lo 反转做空
    mode="trend" (商品实测方向, h10 IC=+0.096 t=+2.3): z>hi 顺势做空偏置
    """
    z = (skew - skew.rolling(z_window).mean()) / skew.rolling(z_window).std()
    sig = pd.Series(0.0, index=skew.index)
    if mode == "reversal":
        sig[z > hi] = 1.0
        sig[z < lo] = -1.0
    elif mode == "trend":
        sig[z > hi] = -1.0
        sig[z < lo] = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return sig


def vrp_confirm_multiplier(
    sig: pd.DataFrame, vrp_positive: pd.DataFrame, same: float = 1.15, diff: float = 0.85
) -> pd.DataFrame:
    """VRP 与信号同向 ×same / 反向 ×diff (品种×日对齐; 无 VRP 数据 → ×1)."""
    aligned_vrp = vrp_positive.reindex(sig.index).reindex(columns=sig.columns)
    same_dir = (sig > 0) & (aligned_vrp == 1)
    diff_dir = (sig < 0) & (aligned_vrp == 1)
    has_vrp = aligned_vrp.notna()
    mult = pd.DataFrame(1.0, index=sig.index, columns=sig.columns)
    mult[same_dir] = same
    mult[diff_dir] = diff
    mult[~has_vrp] = 1.0
    mult[sig == 0] = 1.0
    return mult


def iv_regime_multiplier(
    iv_percentile: pd.DataFrame, hi: float = 0.75, lo: float = 0.25,
    hi_scale: float = 0.7, lo_scale: float = 1.3,
) -> pd.DataFrame:
    """IV 分位 regime 乘数: 高分位压缩 / 低分位放大 (无数据 → ×1)."""
    pct = iv_percentile.reindex(columns=iv_percentile.columns)
    mult = pd.DataFrame(1.0, index=pct.index, columns=pct.columns)
    mult[pct > hi] = hi_scale
    mult[pct < lo] = lo_scale
    return mult


def xsec_z(df: pd.DataFrame) -> pd.DataFrame:
    """逐行截面标准化 (均值0/方差1; 行内有效值<2 → 全 0)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    z = df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    thin = df.notna().sum(axis=1) < 2
    z[thin] = 0.0
    return z


def build_iv_enhanced_signal(
    close: pd.DataFrame,
    iv_panel: pd.DataFrame,
    skew_sig: pd.DataFrame,
    lambda_skew: float = 1.0,
    tier: int = 3,
    weekly: bool = True,
    use_vrp_mult: bool = True,
    use_iv_regime_mult: bool = True,
) -> pd.DataFrame:
    """IV 增强截面信号: composite 排序 + 双乘数.

    close: 品种×日 close 面板 (与 cta_xsec_momentum.compute_mom_score 同口径)
    iv_panel: load_iv_vrp_panel 输出 (长表)
    skew_sig: 品种×日 skew_reversal 三值信号 (缺失品种自动降级)
    """
    rets = close.pct_change(fill_method=None)
    parts = []
    for w in (4, 13, 26, 52):
        parts.append(close.pct_change(w * 5, fill_method=None))
    mom = pd.concat(parts, axis=1).groupby(level=0, axis=1).mean()
    mom = mom.where(rets.notna().cummax().astype(bool))

    z_mom = xsec_z(mom).fillna(0)
    z_skew = xsec_z(skew_sig.reindex(index=mom.index, columns=mom.columns)).fillna(0)
    composite = z_mom + lambda_skew * z_skew

    ranks = composite.rank(axis=1, pct=True)
    sig = pd.DataFrame(0.0, index=composite.index, columns=composite.columns)
    sig[ranks >= 1 - 1 / tier] = 1.0
    sig[ranks <= 1 / tier] = -1.0

    ivp = iv_panel.pivot(
        index="trade_date", columns="product", values="iv_percentile"
    ).reindex(index=mom.index, columns=mom.columns)
    vpp = iv_panel.pivot(
        index="trade_date", columns="product", values="vrp_positive"
    ).reindex(index=mom.index, columns=mom.columns)

    mult = pd.DataFrame(1.0, index=mom.index, columns=mom.columns)
    if use_iv_regime_mult:
        mult = mult * iv_regime_multiplier(ivp)
    if use_vrp_mult:
        mult = mult * vrp_confirm_multiplier(sig, vpp)
    sig = (sig * mult.clip(0.3, 1.5)).clip(-1.5, 1.5)

    n_active = (sig != 0).sum(axis=1).clip(lower=1)
    sig = sig.div(n_active, axis=0)
    if weekly:
        anchors = composite.index.to_series().resample("W-FRI").last().dropna()
        sig = sig.loc[sig.index.isin(anchors)].reindex(sig.index).ffill().fillna(0)
    return sig


def report_xsec_ic(
    score: pd.DataFrame, close: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 21)
) -> pd.DataFrame:
    """截面 IC: score vs 未来 h 日收益 (月频采样, 全期/IS/OOS 分段)."""
    rows = []
    for h in horizons:
        fwd = close.pct_change(h).shift(-h)
        score_al = score.reindex(close.index)
        ics = []
        dates = []
        for dt in score_al.index[::21]:
            if dt not in fwd.index:
                continue
            row = score_al.loc[dt].dropna()
            if row.empty:
                continue
            fut = fwd.loc[dt].dropna()
            common = row.index.intersection(fut.index)
            if len(common) >= 8:
                ics.append(row[common].rank().corr(fut[common].rank()))
                dates.append(dt)
        ic = pd.Series(ics, index=pd.DatetimeIndex(dates)).dropna()
        full = ic.mean()
        t = full / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan
        rows.append({
            "horizon": h,
            "ic_mean": full,
            "ic_t": t,
            "ic_is": ic[ic.index < "2022-01-01"].mean(),
            "ic_oos": ic[ic.index >= "2022-01-01"].mean(),
            "n_months": len(ic),
        })
    return pd.DataFrame(rows).set_index("horizon")
