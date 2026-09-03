#!/usr/bin/env python3
"""cta_open_execution_test.py — T+1执行口径测试: 收盘执行 vs 开盘执行.

背景:
  现有全部子策略NAV用 close-to-close 收益 (信号T日收盘 → T+1日收盘建仓).
  实盘可在 T+1日开盘执行. 本测试量化两种执行口径的差异.

口径定义 (严格对齐):
  close口径: 信号T-2收盘 → T-1收盘建仓 → 持仓[T-1收盘, T收盘]
             pnl[t] = pos[t-1] × (close[t]/close[t-1] - 1)
  open口径:  信号T-2收盘 → T-1开盘建仓 → 持仓[T-1开盘, T开盘]
             pnl[t] = pos[t-1] × (open[t]/open[t-1] - 1)
             即 ret_df = ret_open.shift(1), ret_open[t] = open[t+1]/open[t] - 1

步骤:
  Phase A: 重建close版NAV, 与现有NAV对账 (验证信号复刻正确性)
  Phase B: open版重算 (同信号, 仅换执行收益)
  Phase C: 对比报告 (Sharpe/AnnRet/MaxDD/Calmar + NAV差)

覆盖子策略 (主推组合的子策略层):
  CU溢价TC / ZN库存 / IH50动40×OI10 / T10y OI变化5 / rbi_steel / jjm_coke / ym_oilmeal
"""
from __future__ import annotations

import warnings

import duckdb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import (  # noqa: E402
    REPORT_DIR, rolling_zscore, vol_target_backtest,
)

warnings.filterwarnings("ignore")

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
CMF_DB = "data/cmf/fundamental.duckdb"
IS_END = "2022-01-01"

CON = duckdb.connect(CTA_DB, read_only=True)
CON2 = duckdb.connect(CMF_DB, read_only=True)


def load_smoothed(product: str) -> pd.DataFrame:
    """smoothed主连OHLC (back-adjusted, 与现有NAV同口径)."""
    df = CON.execute(
        f"SELECT trade_date, open, close FROM smoothed_{product.lower()}_daily "
        f"WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")


def load_cffex(tbl: str, label: str) -> pd.DataFrame:
    """cffex合约级数据 (与cta_cross_asset同口径: drop_duplicates keep last)."""
    df = CON.execute(
        f"SELECT trade_date, open, close, open_interest FROM {tbl} "
        f"WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df.rename(columns={"close": label})


def load_vall(product: str) -> pd.DataFrame:
    """v_all_futures合约级 (与cta_cross_commodity同口径)."""
    df = CON.execute(
        f"SELECT trade_date, open, close, open_interest FROM v_all_futures "
        f"WHERE product='{product}' AND close IS NOT NULL AND open IS NOT NULL "
        f"ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df


def make_ret_open(df: pd.DataFrame) -> pd.DataFrame:
    """open-to-open收益: ret_open[t] = open[t+1]/open[t] - 1."""
    return df["open"].shift(-1) / df["open"] - 1


def bt(signal: pd.DataFrame, close: pd.DataFrame, ret_df: pd.DataFrame | None,
        cost_bps: float, target_vol: float, pit_shift: int) -> dict:
    kwargs = dict(cost_bps={signal.columns[0]: cost_bps},
                   target_vol=target_vol, pit_shift=pit_shift)
    if ret_df is not None:
        kwargs["ret_df"] = ret_df.to_frame(signal.columns[0])
    return vol_target_backtest(signal, close, **kwargs)


def nav_stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).fillna(0)
    ann = nav.iloc[-1] ** (252 / len(nav)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"sharpe": ann / (r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            "ann_ret": ann, "max_dd": dd, "nav_end": nav.iloc[-1]}


# ────────────────────────────────────────────────────────────────────────────
# 各子策略: 信号复刻 (精确按原脚本)
# ────────────────────────────────────────────────────────────────────────────

def strat_cu_premium() -> dict:
    """CU溢价TC (cta_cu_premium.py): 4因子IS定方向, tv=0.08, cost=3bp, pit=0."""
    from cta_cu_premium import (load_premium, load_tc, build_factors,
                                 load_cu_close, MIN_IS_SPREAD, Z_EXTREME)
    from cta_shared.factor_research import compute_ic, is_oos_direction_consistency

    close = load_cu_close()
    factors = build_factors(load_premium(), load_tc(), close.index)
    fwd5 = close.pct_change(5, fill_method=None).shift(-5)
    sig = pd.Series(0.0, index=close.index)
    for name, f in factors.items():
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END,
                                             min_is_spread=MIN_IS_SPREAD)
        if cons and cons["agree"] and abs(cons["is_spread"]) > MIN_IS_SPREAD:
            sig += (np.sign(f) * np.sign(cons["is_spread"])
                     * (f.abs() > Z_EXTREME).astype(float)).fillna(0)
    sig = np.sign(sig)
    return {"label": "CU溢价TC", "signal": sig.to_frame("CU"),
             "close": close.to_frame("CU"), "cost": 3.0, "tv": 0.08, "pit": 0,
             "existing": "cu_premium_nav.csv"}


def strat_zn_inventory() -> dict:
    """ZN库存 (cta_zn_inventory.py): LME库存z52极端化, tv=0.08, cost=10bp."""
    from cta_zn_inventory import load_zn_inventory, load_zn_close, compute_zn_signal
    inv = load_zn_inventory()
    close = load_zn_close()
    common = inv.dropna().index.intersection(close.dropna().index)
    signal = compute_zn_signal(inv[common], z_window=52, z_extreme=1.0)
    return {"label": "ZN库存", "signal": signal.to_frame("ZN"),
             "close": close[common].to_frame("ZN"), "cost": 10.0, "tv": 0.08,
             "pit": 0, "existing": "zn_inventory_nav.csv"}


def strat_ih50() -> dict:
    """IH50动40×OI10 (cta_cross_asset.py): tv=0.10, cost=2bp, pit=1."""
    df = load_cffex("cffex_ih_futures_daily", "IH50")
    ret = df["IH50"].pct_change(fill_method=None)
    mom40 = ret.rolling(40).sum()
    oi10 = df["open_interest"].pct_change(10, fill_method=None)
    sig = pd.DataFrame({"IH50": np.sign(mom40 * oi10).fillna(0).values}, index=df.index)
    return {"label": "IH50动40×OI10", "signal": sig, "close": df[["IH50"]],
             "cost": 2.0, "tv": 0.10, "pit": 1, "existing": "ih50_oi_nav.csv",
             "open_series": df["open"]}


def strat_t10y() -> dict:
    """T10y OI变化5 (cta_cross_asset.py): tv=0.10, cost=2bp, pit=1."""
    df = load_cffex("cffex_t_futures_daily", "T10y")
    oi5 = df["open_interest"].pct_change(5, fill_method=None)
    sig = pd.DataFrame({"T10y": np.sign(oi5).fillna(0).values}, index=df.index)
    return {"label": "T10y OI变化5", "signal": sig, "close": df[["T10y"]],
             "cost": 2.0, "tv": 0.10, "pit": 1, "existing": "t10y_oi_nav.csv",
             "open_series": df["open"]}


def _chain(p1: str, p2: str, label: str, existing: str) -> dict:
    """产业链反转 (cta_cross_commodity.py): -z120极端化, tv=0.10, cost=2bp, pit=1."""
    d1, d2 = load_vall(p1), load_vall(p2)
    common = d1.index.intersection(d2.index)
    c1 = d1["close"].reindex(common)
    c2 = d2["close"].reindex(common)
    spread = np.log(c1 / c2)
    roll = spread.rolling(120, min_periods=60)
    z = (spread - roll.mean()) / roll.std()
    sig_raw = -np.sign(z) * (z.abs() > 1.0).astype(float)
    sig = pd.DataFrame({label: sig_raw.fillna(0).values}, index=common)
    close = pd.DataFrame({label: c1.values}, index=common)
    return {"label": label, "signal": sig, "close": close, "cost": 2.0,
             "tv": 0.10, "pit": 1, "existing": existing,
             "open_p1": d1["open"].reindex(common), "open_p2": d2["open"].reindex(common)}


def strat_chain_all() -> list[dict]:
    return [_chain("RB", "I", "rbi_steel", "chain_rbi_steel_nav.csv"),
             _chain("J", "JM", "jjm_coke", "chain_jjm_coke_nav.csv"),
             _chain("Y", "M", "ym_oilmeal", "chain_ym_oilmeal_nav.csv")]


# ────────────────────────────────────────────────────────────────────────────

def run_one(s: dict) -> dict:
    """单子策略: close版对账 + open版重算."""
    signal, close = s["signal"], s["close"]
    existing_nav = pd.read_csv(REPORT_DIR / s["existing"],
                                index_col=0, parse_dates=True).iloc[:, 0]

    # Phase A: close版对账 (ret_df=None → close.pct_change, 原逻辑)
    res_close = bt(signal, close, None, s["cost"], s["tv"], s["pit"])
    nav_close = res_close["nav"]
    common = nav_close.index.intersection(existing_nav.index)
    diff = (nav_close.reindex(common) - existing_nav.reindex(common)).abs().max()
    recon = "PASS" if diff < 1e-6 else f"DIFF={diff:.4f}"

    # Phase B: open版 (ret_df = ret_open.shift(1))
    # open必须与close同表同合约口径, 否则换月日两表错位会污染收益
    if "open_series" in s:  # cffex口径: 同表open
        ret_open = s["open_series"].shift(-1) / s["open_series"] - 1
    elif "open_p1" in s:  # 产业链: 用p1品种的open (与close同品种)
        ret_open = s["open_p1"].shift(-1) / s["open_p1"] - 1
    else:  # smoothed口径: 同表open
        df_ohlc = load_smoothed(str(signal.columns[0]).lower())
        ret_open = make_ret_open(df_ohlc)
    ret_open_aligned = ret_open.reindex(close.index)
    res_open = bt(signal, close, ret_open_aligned.shift(1), s["cost"], s["tv"], s["pit"])
    nav_open = res_open["nav"]

    st_close, st_open = nav_stats(nav_close), nav_stats(nav_open)
    st_exist = nav_stats(existing_nav)
    return {
        "子策略": s["label"], "对账": recon,
        "现有Sharpe": st_exist["sharpe"], "重建Sharpe(close)": st_close["sharpe"],
        "open_Sharpe": st_open["sharpe"],
        "close_AnnRet": st_close["ann_ret"], "open_AnnRet": st_open["ann_ret"],
        "close_MaxDD": st_close["max_dd"], "open_MaxDD": st_open["max_dd"],
        "close_Calmar": st_close["ann_ret"] / abs(st_close["max_dd"]),
        "open_Calmar": st_open["ann_ret"] / abs(st_open["max_dd"]),
        "_nav_close": nav_close, "_nav_open": nav_open,
    }


def main():
    print("=" * 110)
    print("T+1执行口径测试: 收盘执行(close-to-close) vs 开盘执行(open-to-open)")
    print("口径: close=T-1收盘建仓→T收盘; open=T-1开盘建仓→T开盘")
    print("=" * 110)

    strategies = [strat_cu_premium(), strat_zn_inventory(), strat_ih50(),
                   strat_t10y()] + strat_chain_all()

    rows = []
    navs_close, navs_open = {}, {}
    for s in strategies:
        r = run_one(s)
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        navs_close[r["子策略"]] = r["_nav_close"]
        navs_open[r["子策略"]] = r["_nav_open"]
        print(f"  {r['子策略']:<16} 对账={r['对账']:<12} "
              f"close Sharpe={r['重建Sharpe(close)']:+.3f} → open Sharpe={r['open_Sharpe']:+.3f}")

    df = pd.DataFrame(rows).set_index("子策略")
    cols = ["对账", "现有Sharpe", "重建Sharpe(close)", "open_Sharpe",
             "close_AnnRet", "open_AnnRet", "close_MaxDD", "open_MaxDD",
             "close_Calmar", "open_Calmar"]
    print("\n=== 汇总 ===")
    print(df[cols].round(4).to_string())
    df.drop(columns=["对账"]).to_csv(REPORT_DIR / "open_execution_test.csv")

    # 等权合成对比 (子策略层整体)
    print("\n=== 子策略等权合成 (close vs open) ===")
    for name, navs in [("close等权", navs_close), ("open等权", navs_open)]:
        rets = pd.DataFrame({k: v.pct_change(fill_method=None).fillna(0)
                              for k, v in navs.items()}).dropna()
        combo = rets.mean(axis=1)
        st = nav_stats((1 + combo).cumprod())
        print(f"  {name:<10} Sharpe={st['sharpe']:+.3f} AnnRet={st['ann_ret']:+.2%} "
              f"MaxDD={st['max_dd']:.2%}")

    CON.close()
    CON2.close()


if __name__ == "__main__":
    main()
