#!/usr/bin/env python3
"""cta_cross_commodity.py — 跨商品品种(产业链+板块)子策略研究.

子策略:
  - 板块动量+OI: agri动20×OI5 (OOS=1.34, IS反向, regime change)
  - 产业链价差反转:
    - rbi_steel (RB/I) - 钢厂利润 z-score反转, OOS=0.32
    - jjm_coke (J/JM) - 焦化利润 z-score反转, OOS=0.31
    - ym_oilmeal (Y/M) - 油粕比 z-score反转, OOS=0.22

PIT: shift(1) (产业链对数价差→次日信号)

输出:
  - reports/cta/{chain_*_nav.csv, sector_agri_nav.csv}
  - reports/cta/cross_commodity_combination.csv
"""
from __future__ import annotations

import warnings

import duckdb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import (  # noqa: E402
    vol_target_backtest, period_stats, REPORT_DIR,
)

warnings.filterwarnings("ignore")
DB_PATH = "/home/ll/Public/DataCache/futures_options_daily.duckdb"


def load_close(product: str) -> pd.DataFrame:
    """加载品种close+OI."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT trade_date, close, open_interest FROM v_all_futures "
        f"WHERE product='{product}' AND close IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df


def build_chain_signal(p1: str, p2: str, label: str) -> dict:
    """产业链价差反转: -z * (|z|>1).

    p1/p2: 上下游品种
    signal = -sign(z(spread)) if |z|>1 else 0
    spread = log(c1/c2)
    """
    d1 = load_close(p1)
    d2 = load_close(p2)
    common = d1.index.intersection(d2.index)
    c1 = d1["close"].reindex(common)
    c2 = d2["close"].reindex(common)
    spread = np.log(c1 / c2)
    roll = spread.rolling(120, min_periods=60)
    z = (spread - roll.mean()) / roll.std()
    sig_raw = -np.sign(z) * (z.abs() > 1.0).astype(float)
    sig = pd.DataFrame({label: sig_raw.fillna(0).values}, index=common)
    close = pd.DataFrame({label: c1.values}, index=common)
    res = vol_target_backtest(sig, close, cost_bps={label: 2.0},
                                target_vol=0.10, pit_shift=1)
    res["nav"].to_csv(REPORT_DIR / f"chain_{label}_nav.csv", header=["nav"])
    st = period_stats(res["pnl"])
    print(f"{label} ({p1}/{p2}): Sharpe={res['sharpe']:+.3f} "
          f"IS={st['is']['sharpe']:+.3f} OOS={st['oos']['sharpe']:+.3f} "
          f"DD={res['max_dd']:.2%}")
    return res


def build_sector_agri_signal() -> dict:
    """agri板块动量+OI5因子 (20日动量×5日OI变化)."""
    prods = ['A', 'B', 'C', 'CS', 'M', 'RM', 'Y', 'P', 'OI', 'CF', 'SR',
             'WH', 'PM', 'RI', 'LR', 'JR', 'JD', 'AP', 'CJ', 'CY', 'PK']
    closes, ois = {}, {}
    for p in prods:
        df = load_close(p)
        if len(df) > 1000:
            closes[p] = df["close"]
            ois[p] = df["open_interest"]
    close_df = pd.DataFrame(closes).sort_index().ffill()
    oi_df = pd.DataFrame(ois).sort_index().ffill()
    proxy_close = close_df.iloc[:, [0]].rename(columns={close_df.columns[0]: "PROXY"})
    sector_ret = close_df.pct_change(fill_method=None).mean(axis=1).fillna(0)
    sector_oi5 = oi_df.mean(axis=1).pct_change(5, fill_method=None).fillna(0)
    mom = sector_ret.rolling(20).sum()
    sig_raw = np.sign(mom * sector_oi5).fillna(0)
    sig = pd.DataFrame({"PROXY": sig_raw.values}, index=proxy_close.index)
    res = vol_target_backtest(sig, proxy_close, cost_bps={"PROXY": 2.0},
                                target_vol=0.10, pit_shift=1)
    res["nav"].to_csv(REPORT_DIR / "sector_agri_nav.csv", header=["nav"])
    st = period_stats(res["pnl"])
    print(f"agri动20×OI5: Sharpe={res['sharpe']:+.3f} "
          f"IS={st['is']['sharpe']:+.3f} OOS={st['oos']['sharpe']:+.3f}")
    return res


def combination_experiments() -> pd.DataFrame:
    """合并实验: 主CTA+ZN+CU+IH50+T10y+rbi+jjm+ym+agri."""
    main_nav = pd.read_csv(REPORT_DIR / "final_nav_lump.csv",
                            index_col=0, parse_dates=True).iloc[:, 0]
    zn_nav = pd.read_csv(REPORT_DIR / "zn_inventory_nav.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    cu_nav = pd.read_csv(REPORT_DIR / "cu_premium_nav.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    ih_nav = pd.read_csv(REPORT_DIR / "ih50_oi_nav.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    t10y_nav = pd.read_csv(REPORT_DIR / "t10y_oi_nav.csv",
                            index_col=0, parse_dates=True).iloc[:, 0]
    rbi_nav = pd.read_csv(REPORT_DIR / "chain_rbi_steel_nav.csv",
                           index_col=0, parse_dates=True).iloc[:, 0]
    jjm_nav = pd.read_csv(REPORT_DIR / "chain_jjm_coke_nav.csv",
                           index_col=0, parse_dates=True).iloc[:, 0]
    ym_nav = pd.read_csv(REPORT_DIR / "chain_ym_oilmeal_nav.csv",
                          index_col=0, parse_dates=True).iloc[:, 0]
    agri_nav = pd.read_csv(REPORT_DIR / "sector_agri_nav.csv",
                             index_col=0, parse_dates=True).iloc[:, 0]

    rets = pd.DataFrame({
        "主CTA": main_nav.pct_change(fill_method=None).fillna(0),
        "ZN库存": zn_nav.pct_change(fill_method=None).fillna(0),
        "CU溢价TC": cu_nav.pct_change(fill_method=None).fillna(0),
        "IH50": ih_nav.pct_change(fill_method=None).fillna(0),
        "T10y": t10y_nav.pct_change(fill_method=None).fillna(0),
        "rbi_steel": rbi_nav.pct_change(fill_method=None).fillna(0),
        "jjm_coke": jjm_nav.pct_change(fill_method=None).fillna(0),
        "ym_oilmeal": ym_nav.pct_change(fill_method=None).fillna(0),
        "agri动20×OI5": agri_nav.pct_change(fill_method=None).fillna(0),
    }).dropna()

    IS_END = "2022-01-01"

    def stats(s: pd.Series) -> dict:
        nav = (1 + s).cumprod()
        ann = nav.iloc[-1] ** (252 / len(s)) - 1
        vol = s.std() * np.sqrt(252)
        dd = (nav / nav.cummax() - 1).min()
        return {"sharpe": ann / vol if vol > 0 else 0, "ann_ret": ann, "max_dd": dd}

    portfolios = [
        # 已用 (主CTA, ZN, CU, IH50, T10y, rbi, jjm, ym, agri)
        (1.0, 0, 0, 0, 0, 0, 0, 0, 0, "主CTA单独"),
        (0.60, 0.05, 0.15, 0.10, 0.10, 0, 0, 0, 0, "原跨资产最优"),
        (0.55, 0.05, 0.15, 0.10, 0.10, 0.05, 0, 0, 0, "+rbi5%"),
        (0.55, 0.05, 0.15, 0.10, 0.10, 0, 0.05, 0, 0, "+jjm5%"),
        (0.55, 0.05, 0.15, 0.10, 0.10, 0, 0, 0.05, 0, "+ym5%"),
        (0.55, 0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0, 0, "+rbi+jjm各5%"),
        (0.50, 0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0, 0, "+rbi+jjm5+主CTA50"),
        (0.45, 0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0, "+3产业链+主CTA45"),
        (0.50, 0.05, 0.10, 0.05, 0.05, 0.10, 0.10, 0.05, 0, "分散到8策略"),
        (0.50, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, "+agri10%(试验)"),
        (0.45, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, "+agri15%(试验)"),
    ]

    print("\n=== 跨商品产业链合并实验 ===")
    print(f"{'组合':<30} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>8} | "
          f"{'IS':>6} {'OOS':>6} {'OOS-AR':>8}")
    print("-" * 100)

    rows = []
    for p in portfolios:
        if abs(sum(p[:-1]) - 1.0) > 0.01:
            continue
        w_main, w_zn, w_cu, w_ih, w_t, w_rbi, w_jjm, w_ym, w_agri, label = p
        combo = (w_main * rets["主CTA"] + w_zn * rets["ZN库存"]
                  + w_cu * rets["CU溢价TC"] + w_ih * rets["IH50"]
                  + w_t * rets["T10y"] + w_rbi * rets["rbi_steel"]
                  + w_jjm * rets["jjm_coke"] + w_ym * rets["ym_oilmeal"]
                  + w_agri * rets["agri动20×OI5"])
        full = stats(combo)
        is_s = stats(combo[combo.index < IS_END])
        oos_s = stats(combo[combo.index >= IS_END])
        print(f"{label:<30} {full['sharpe']:>7.3f} {full['ann_ret']:>7.2%} "
              f"{full['max_dd']:>8.2%} | {is_s['sharpe']:>6.3f} "
              f"{oos_s['sharpe']:>6.3f} {oos_s['ann_ret']:>7.2%}")
        rows.append({
            "组合": label, "主CTA": w_main, "ZN": w_zn, "CU": w_cu,
            "IH50": w_ih, "T10y": w_t, "rbi": w_rbi, "jjm": w_jjm,
            "ym": w_ym, "agri": w_agri, **full,
            "IS_Sharpe": is_s["sharpe"], "OOS_Sharpe": oos_s["sharpe"],
            "OOS_AnnRet": oos_s["ann_ret"],
        })

    df = pd.DataFrame(rows)
    out = REPORT_DIR / "cross_commodity_combination.csv"
    df.to_csv(out, index=False)
    print(f"\n保存: {out}")
    return df


def main():
    print("=== 产业链子策略 ===")
    build_chain_signal("RB", "I", "rbi_steel")
    build_chain_signal("J", "JM", "jjm_coke")
    build_chain_signal("Y", "M", "ym_oilmeal")

    print("\n=== 板块子策略 ===")
    build_sector_agri_signal()

    df = combination_experiments()
    print("\n=== 推荐 (全期Sharpe最高, 不含agri试验) ===")
    valid = df[~df["组合"].str.contains("agri")]
    best = valid.loc[valid["sharpe"].idxmax()]
    print(f"  {best['组合']}: Sharpe={best['sharpe']:.3f} "
          f"MaxDD={best['max_dd']:.2%} OOS={best['OOS_Sharpe']:.3f}")


if __name__ == "__main__":
    main()
