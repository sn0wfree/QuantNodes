#!/usr/bin/env python3
"""cta_cross_asset.py — 跨资产因子研究与合并实验.

子策略:
  - IH50动40×OI10: 上证50 40日动量 × 10日OI变化方向一致时持仓
  - T10y OI变化5:  国债10年 5日持仓量增加→做多

PIT: shift(1) (跨资产日频数据,盘后发布 → T+1可用)

输出:
  - reports/cta/{ih50_oi_nav.csv, t10y_oi_nav.csv, cross_asset_combination.csv}
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


def load_data(tbl: str, label: str) -> pd.DataFrame:
    """加载跨资产品种close+open_interest."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT trade_date, close, open_interest, volume FROM {tbl} "
        f"WHERE close IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df.rename(columns={"close": label})


def build_ih50_oi_signal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IH50动40×OI10: 40日动量 × 10日OI变化方向一致时持仓."""
    df = load_data("cffex_ih_futures_daily", "IH50")
    close = df[["IH50"]]
    ret = close["IH50"].pct_change(fill_method=None)
    mom40 = ret.rolling(40).sum()
    oi10 = df["open_interest"].pct_change(10, fill_method=None)
    sig = pd.DataFrame({"IH50": np.sign(mom40 * oi10).fillna(0).values}, index=close.index)
    return sig, close, pd.DataFrame({"open_interest": df["open_interest"]})


def build_t10y_oi_signal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """T10y OI变化5: 5日OI增加→做多."""
    df = load_data("cffex_t_futures_daily", "T10y")
    close = df[["T10y"]]
    oi5 = df["open_interest"].pct_change(5, fill_method=None)
    sig = pd.DataFrame({"T10y": np.sign(oi5).fillna(0).values}, index=close.index)
    return sig, close, pd.DataFrame({"open_interest": df["open_interest"]})


def evaluate(sig: pd.DataFrame, close: pd.DataFrame, label: str,
              cost_bps: float = 2.0, target_vol: float = 0.10,
              pit_shift: int = 1) -> dict:
    res = vol_target_backtest(
        sig, close, cost_bps={label: cost_bps}, target_vol=target_vol, pit_shift=pit_shift,
    )
    st = period_stats(res["pnl"])
    print(f"\n=== {label} ===")
    print(f"  全期: Sharpe={res['sharpe']:+.3f} AR={res['ann_ret']:+.2%} DD={res['max_dd']:.2%}")
    turn_str = f"{res['turnover_annual']:.1f}"
    print(f"  IS: {st['is']['sharpe']:+.3f}  OOS: {st['oos']['sharpe']:+.3f}  Turnover={turn_str}x")
    return {**res, **st}


def combination_experiments() -> pd.DataFrame:
    """合并实验: 主CTA+ZN+CU+IH50+T10y."""
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

    main_ret = main_nav.pct_change(fill_method=None).fillna(0).rename("主CTA")
    zn_ret = zn_nav.pct_change(fill_method=None).fillna(0).rename("ZN库存")
    cu_ret = cu_nav.pct_change(fill_method=None).fillna(0).rename("CU溢价TC")
    ih_ret = ih_nav.pct_change(fill_method=None).fillna(0).rename("IH50动40×OI10")
    t10y_ret = t10y_nav.pct_change(fill_method=None).fillna(0).rename("T10y OI变化5")

    common = main_ret.index
    for r in [zn_ret, cu_ret, ih_ret, t10y_ret]:
        common = common.intersection(r.index)
    rets = pd.concat([
        main_ret.reindex(common), zn_ret.reindex(common), cu_ret.reindex(common),
        ih_ret.reindex(common), t10y_ret.reindex(common),
    ], axis=1).fillna(0)

    IS_END = "2022-01-01"

    def stats(s: pd.Series) -> dict:
        nav = (1 + s).cumprod()
        ann = nav.iloc[-1] ** (252 / len(s)) - 1
        vol = s.std() * np.sqrt(252)
        dd = (nav / nav.cummax() - 1).min()
        return {"sharpe": ann / vol if vol > 0 else 0, "ann_ret": ann, "max_dd": dd}

    portfolios = [
        (1.0, 0, 0, 0, 0, "主CTA单独"),
        (0.80, 0.05, 0.15, 0, 0, "原最优 80/5/15/0/0"),
        (0.75, 0.05, 0.15, 0.05, 0, "+IH5%"),
        (0.70, 0.05, 0.15, 0.10, 0, "+IH10%"),
        (0.65, 0.05, 0.15, 0.15, 0, "+IH15%"),
        (0.75, 0.05, 0.15, 0, 0.05, "+T10y5%"),
        (0.70, 0.05, 0.15, 0.05, 0.05, "+IH+T10y各5%"),
        (0.65, 0.05, 0.15, 0.10, 0.05, "+IH10+T10y5"),
        (0.60, 0.05, 0.15, 0.15, 0.05, "+IH15+T10y5"),
        (0.70, 0.05, 0.10, 0.10, 0.05, "IH10+T10y5+CU10+ZN5"),
        (0.60, 0.05, 0.15, 0.10, 0.10, "IH10+T10y10+主CTA60"),
    ]

    print("\n=== 跨资产合并实验 ===")
    print(f"{'组合':<35} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>8} | "
          f"{'IS':>6} {'OOS':>6} {'OOS-AR':>8}")
    print("-" * 100)

    rows = []
    for w_main, w_zn, w_cu, w_ih, w_t, label in portfolios:
        if abs(w_main + w_zn + w_cu + w_ih + w_t - 1.0) > 0.01:
            continue
        combo = (w_main * rets["主CTA"] + w_zn * rets["ZN库存"] + w_cu * rets["CU溢价TC"]
                  + w_ih * rets["IH50动40×OI10"] + w_t * rets["T10y OI变化5"])
        full = stats(combo)
        is_s = stats(combo[combo.index < IS_END])
        oos_s = stats(combo[combo.index >= IS_END])
        print(f"{label:<35} {full['sharpe']:>7.3f} {full['ann_ret']:>7.2%} "
              f"{full['max_dd']:>8.2%} | {is_s['sharpe']:>6.3f} "
              f"{oos_s['sharpe']:>6.3f} {oos_s['ann_ret']:>7.2%}")
        rows.append({
            "组合": label, "主CTA": w_main, "ZN": w_zn, "CU": w_cu,
            "IH50": w_ih, "T10y": w_t, **full,
            "IS_Sharpe": is_s["sharpe"], "OOS_Sharpe": oos_s["sharpe"],
            "OOS_AnnRet": oos_s["ann_ret"],
        })

    df = pd.DataFrame(rows)
    out = REPORT_DIR / "cross_asset_combination.csv"
    df.to_csv(out, index=False)
    print(f"\n保存: {out}")
    return df


def main():
    # IH50 动40×OI10
    sig_ih, close_ih, _ = build_ih50_oi_signal()
    res_ih = evaluate(sig_ih, close_ih, "IH50", cost_bps=2.0)
    res_ih["nav"].to_csv(REPORT_DIR / "ih50_oi_nav.csv", header=["nav"])

    # T10y OI变化5
    sig_t, close_t, _ = build_t10y_oi_signal()
    res_t = evaluate(sig_t, close_t, "T10y", cost_bps=2.0)
    res_t["nav"].to_csv(REPORT_DIR / "t10y_oi_nav.csv", header=["nav"])

    # 合并实验
    df = combination_experiments()
    print("\n=== 推荐组合 (全期Sharpe最高) ===")
    best = df.loc[df["sharpe"].idxmax()]
    print(f"  {best['组合']}: Sharpe={best['sharpe']:.3f} MaxDD={best['max_dd']:.2%}")


if __name__ == "__main__":
    main()
