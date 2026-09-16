# coding=utf-8
"""Stage 2-A: 期权凸性执行层 — 主 CTA 趋势的凸性放大器 (overlay).

定位 (A 路线定稿, 2026-09-16):
    截面动量 alpha 已证伪 (见 docs/cta/xsec_option_iv_ablation.md);
    期权池基础设施转用于: 跟随主 CTA 方向, 用期权费换取趋势凸性.

模型 (delta 线性化, greeks 实测值, 无拍脑袋参数):
    daily_pnl_t = Σ_i pos_i,t × (delta_i,t × r_i,t − theta_i,t / 252)
    pos = dir × mult × (THETA_BUDGET/252 / 当日总theta), 年预算 2%

结构选择 (Q6 决策, HV20 分位判定, build/回测共用 struct_flag):
    flag=1 OTM买入腿: HV分位 < hv_lo, delta∈[0.25,0.45] 代表合约
    flag=2 价差组合:  HV分位 >= hv_lo, long δ∈[0.35,0.55] − short δ∈[0.10,0.20]
    flag=0 无期权/无信号/theta无效

乘数: iv_regime (IV分位<0.3 → ×1.3, >0.75 → ×0.5, 无数据 → ×1.0)
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import CTA_DB

OPTION_UNIVERSE = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU"]
DELTA_EQUIV = 0.30
PREM_BUDGET = 0.02
TTM_MAX = 30 / 365.0
DELTA_OTM = (0.25, 0.45)
DELTA_LONG = (0.35, 0.55)
DELTA_SHORT = (0.10, 0.20)
IV_LO, IV_HI = 0.3, 0.75
IV_LO_SCALE, IV_HI_SCALE = 1.3, 0.5


def load_hv20_percentile(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """品种×日 HV20 + 252日滚动分位 (volatility_hv_daily)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    try:
        df = con.execute(
            f"SELECT product, trade_date, hv_20 FROM volatility_hv_daily "
            f"WHERE trade_date >= '{start}' AND trade_date <= '{end}'"
        ).df()
    finally:
        con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wide = df.pivot(index="trade_date", columns="product", values="hv_20").sort_index()
    pct = wide.rolling(252, min_periods=60).apply(
        lambda x: float((x.iloc[-1] > x).mean()), raw=False
    )
    return wide, pct


def load_premium_map(products: list[str], start: str, end: str) -> pd.DataFrame:
    """合约级 close 映射 (premium_frac = close/spot 用, cache 未存 close)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    try:
        pl = ",".join(f"'{p}'" for p in products)
        df = con.execute(
            f"SELECT product, symbol, trade_date, close FROM v_all_options "
            f"WHERE product IN ({pl}) AND trade_date >= '{start}' "
            f"AND trade_date <= '{end}'"
        ).df()
    finally:
        con.close()
    df["symbol"] = df["symbol"].str.upper()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates(["product", "symbol", "trade_date"])


def pick_struct_from_cache(
    cache: pd.DataFrame, ttm_max: float = TTM_MAX
) -> pd.DataFrame:
    """从 greeks cache + premium map 聚合品种×日结构参数.

    premium_frac = 合约 close / spot; theta_frac = premium_frac / 剩余天数
    (BS 反解 theta 数值不稳, 用权利金线性衰减近似, 30d 持期口径).
    返回列: trade_date, product, d_otm, prem_otm, d_spread, prem_spread, ttm_days
    """
    w = cache[cache["valid"]].copy()
    w["trade_date"] = pd.to_datetime(w["trade_date"])
    w["expiry_date"] = pd.to_datetime(w["expiry_date"])
    w["ttm_days"] = (w["expiry_date"] - w["trade_date"]).dt.days
    w = w[(w["ttm_days"] > 0) & (w["ttm_days"] <= ttm_max * 365)]
    w["premium_frac"] = w["close"] / w["spot"]
    w = w[(w["premium_frac"] > 0) & (w["premium_frac"] <= 0.5)]
    w["delta_abs"] = w["delta"].abs()

    rows = []
    for (p, dt), g in w.groupby(["product", "trade_date"]):
        otm = g[(g["delta_abs"] >= DELTA_OTM[0]) & (g["delta_abs"] <= DELTA_OTM[1])]
        legs_long = g[
            (g["delta_abs"] >= DELTA_LONG[0]) & (g["delta_abs"] <= DELTA_LONG[1])
        ]
        legs_short = g[
            (g["delta_abs"] >= DELTA_SHORT[0]) & (g["delta_abs"] <= DELTA_SHORT[1])
        ]
        if otm.empty or legs_long.empty or legs_short.empty:
            continue
        prem_spread = (
            legs_long["premium_frac"].mean() - legs_short["premium_frac"].mean()
        )
        if prem_spread <= 0:
            continue
        rows.append({
            "product": p, "trade_date": dt,
            "d_otm": float(otm["delta"].mean()),
            "prem_otm": float(otm["premium_frac"].mean()),
            "ttm_days": float(g["ttm_days"].median()),
            "d_spread": float(legs_long["delta"].mean()
                              - legs_short["delta"].mean()),
            "prem_spread": float(prem_spread),
        })
    return pd.DataFrame(rows)


def build_option_convexity_positions(
    signals: pd.DataFrame,
    struct: pd.DataFrame,
    hv_pct: pd.DataFrame,
    iv_pct: pd.DataFrame | None = None,
    hv_lo: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """期权凸性名义 delta 敞口 (品种×日, NAV 比例).

    delta_notional_i = dir_i × mult_i × DELTA_EQUIV / n_active
    (overlay 替代语义: 期权腿提供 delta_equiv 比例 NAV 的方向敞口,
    premium 成本自动 = 敞口 × prem/(delta × ttm), 年化 ≤ PREM_BUDGET 检验)

    返回:
        (pos, flag)
        pos: 品种×日名义 delta 敞口 (NAV 比例)
        flag: 品种×日结构标记 {0: 无, 1: OTM买入, 2: 价差组合}
    """
    idx = signals.index
    cols = [c for c in OPTION_UNIVERSE if c in signals.columns]
    st = struct.set_index(["trade_date", "product"])
    hvp = hv_pct.reindex(index=idx, columns=cols)
    ivp = (
        iv_pct.reindex(index=idx, columns=cols)
        if iv_pct is not None
        else pd.DataFrame(np.nan, index=idx, columns=cols)
    )

    pos = pd.DataFrame(0.0, index=idx, columns=cols)
    flag = pd.DataFrame(0, index=idx, columns=cols)

    for dt in idx:
        elig = [c for c in cols if signals.at[dt, c] != 0
                and (dt, c) in st.index]
        if not elig:
            continue
        n_act = len(elig)
        for c in elig:
            dir_i = signals.at[dt, c]
            hv_i = hvp.at[dt, c]
            hv_i = hv_lo if pd.isna(hv_i) else hv_i
            flag.at[dt, c] = 1 if hv_i < hv_lo else 2
            pos.at[dt, c] = dir_i * DELTA_EQUIV / n_act

    mult = pd.DataFrame(1.0, index=idx, columns=cols)
    mult[ivp < IV_LO] = IV_LO_SCALE
    mult[ivp > IV_HI] = IV_HI_SCALE
    mult[ivp.isna()] = 1.0
    pos = pos * mult
    return pos, flag


def run_option_overlay_bt(
    pos: pd.DataFrame, flag: pd.DataFrame, close: pd.DataFrame,
    struct: pd.DataFrame,
) -> dict:
    """期权 overlay 回测 (delta 线性化 + premium 杠杆).

    daily_pnl_t = Σ_i pos_i,t × (delta_i,t × r_i,t − theta_i,t)
    pos = dir × mult × b_i/prem_i; 预算消耗隐含在 (−theta × b/prem) 项.
    delta/theta/prem 按 flag 回查 struct.
    """
    idx = pos.index
    rets = close.pct_change(fill_method=None).reindex(idx)
    st = struct.set_index(["trade_date", "product"])

    pnl = pd.Series(0.0, index=idx)
    prem_used = pd.Series(0.0, index=idx)
    n_active = pd.Series(0, index=idx)

    for dt in idx:
        day_pnl = 0.0
        day_prem = 0.0
        n_act = 0
        for c in pos.columns:
            p = pos.at[dt, c]
            if abs(p) < 1e-12 or flag.at[dt, c] == 0 or (dt, c) not in st.index:
                continue
            g = st.loc[(dt, c)]
            if flag.at[dt, c] == 1:
                delta_i, prem_i = g["d_otm"], g["prem_otm"]
            else:
                delta_i, prem_i = g["d_spread"], g["prem_spread"]
            ttm_days = max(g["ttm_days"], 1.0)
            theta_i = prem_i / ttm_days
            r = rets.at[dt, c] if not pd.isna(rets.at[dt, c]) else 0.0
            day_pnl += p * (delta_i * r - theta_i)
            day_prem += abs(p) * prem_i
            n_act += 1
        pnl.loc[dt] = day_pnl
        prem_used.loc[dt] = day_prem
        n_active.loc[dt] = n_act

    nav = (1 + pnl).cumprod()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0.0
    return {
        "pnl": pnl, "nav": nav, "prem_used": prem_used, "n_active": n_active,
        "sharpe": sharpe,
        "ann_ret": nav.iloc[-1] ** (252 / len(pnl)) - 1,
        "max_dd": (nav / nav.cummax() - 1).min(),
        "prem_yearly": prem_used.mean() * 252,
    }
