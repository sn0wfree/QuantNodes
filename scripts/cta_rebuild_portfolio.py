#!/usr/bin/env python3
"""cta_rebuild_portfolio.py — 数据口径审计后的组合重构测试.

背景 (cta_open_execution_test.py 审计结论):
  - IH50 / rbi_steel 为随机合约拼接伪影 → 止损剔除
  - jjm_coke 被低估 (真实0.540 > 拼接口径0.185)
  - 真实执行口径: 商品(有夜盘) open差异-0.04轻微; T10y -0.03
  - 主CTA数据口径干净 (gradual_continuous), 保持close口径+文档披露

重构组合 (主CTA不变, 子策略层真实口径重建):
  权重方案:
    R0: 55%主CTA×动态杠杆 + CU 45%               (单子策略极简)
    R1: 55%主CTA×动态杠杆 + CU 31.5% + T10y 13.5% (T10y替代IH50)
    R2: 55%主CTA×动态杠杆 + CU 25% + T10y 10% + jjm 10%
    R3: 55%主CTA×动态杠杆 + CU 22.5% + T10y 11.25% + jjm 11.25%
  风控: 动态杠杆[1.0,1.5] + 渐进A自适应(20×40×60, cut0.5) + 5bp换手成本

执行口径: 子策略用 open-to-open (真实执行) 为主, close口径对照.

输出:
  - reports/cta/rebuild_portfolio.csv  组合重构对比表
  - reports/cta/rebuild_subnav_*.csv   真实口径子策略NAV (close/open)
"""
from __future__ import annotations

import warnings

import duckdb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import (  # noqa: E402
    REPORT_DIR, rolling_zscore, is_oos_direction_consistency,
)
from cta_shared.portfolio_tools import (  # noqa: E402
    multi_window_factors, dynamic_leverage, levered_portfolio, turnover_cost,
    DEFAULT_LEV_COST,
)

warnings.filterwarnings("ignore")

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
CMF_DB = "data/cmf/fundamental.duckdb"
IS_END = "2022-01-01"
W_MAIN = 0.55
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0

CON = duckdb.connect(CTA_DB, read_only=True)
CON2 = duckdb.connect(CMF_DB, read_only=True)


def load_smoothed_full(tbl: str) -> pd.DataFrame:
    df = CON.execute(
        f"SELECT trade_date, open, close, open_interest FROM smoothed_{tbl}_daily "
        f"WHERE close IS NOT NULL AND open IS NOT NULL ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.drop_duplicates("trade_date", keep="last").set_index("trade_date")


# ── 子策略重建 (全部smoothed真主连口径) ─────────────────────────────────────

def build_cu():
    """CU溢价TC: 4因子IS定方向 (tfapi溢价+TC, smoothed_cu)."""
    prem = CON2.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name='洋山铜溢价' ORDER BY trade_date").df()
    tc = CON2.execute(
        "SELECT trade_date, value FROM tfapi_indicators "
        "WHERE index_name LIKE '%粗炼费%' ORDER BY trade_date").df()
    for d in (prem, tc):
        d["trade_date"] = pd.to_datetime(d["trade_date"])
    prem = prem.set_index("trade_date")["value"].sort_index()
    tc = tc.set_index("trade_date")["value"].sort_index()

    df = load_smoothed_full("cu")
    close = df["close"]
    cn_days = close.index
    prem_d = prem.reindex(cn_days.union(prem.index)).ffill().reindex(cn_days)
    tc_d = tc.reindex(cn_days.union(tc.index)).ffill().reindex(cn_days)
    factors = {
        "P1": rolling_zscore(prem_d, 252),
        "P2": rolling_zscore(prem_d.pct_change(20), 252),
        "T1": rolling_zscore(tc_d, 252),
        "T2": rolling_zscore(tc_d.pct_change(20), 252),
    }
    factors = {k: v.shift(1) for k, v in factors.items()}  # PIT
    fwd5 = close.pct_change(5, fill_method=None).shift(-5)
    sig = pd.Series(0.0, index=cn_days)
    for name, f in factors.items():
        cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END, min_is_spread=0.05)
        if cons and cons["agree"] and abs(cons["is_spread"]) > 0.05:
            sig += (np.sign(f) * np.sign(cons["is_spread"])
                     * (f.abs() > 1.0).astype(float)).fillna(0)
    sig = np.sign(sig).to_frame("CU")
    return sig, close.to_frame("CU"), df["open"], 3.0, 0.08, 0  # pit=0 (因子自带shift1)


def build_t10y():
    """T10y OI变化5: smoothed_t OI5日变化."""
    df = load_smoothed_full("t")
    oi5 = df["open_interest"].pct_change(5, fill_method=None)
    sig = pd.DataFrame({"T10y": np.sign(oi5).fillna(0).values}, index=df.index)
    return sig, df[["close"]].rename(columns={"close": "T10y"}), df["open"], 2.0, 0.10, 1


def build_jjm():
    """焦化利润反转: log(J/JM) z120 反转."""
    d1, d2 = load_smoothed_full("j"), load_smoothed_full("jm")
    common = d1.index.intersection(d2.index)
    c1, c2 = d1["close"].reindex(common), d2["close"].reindex(common)
    spread = np.log(c1 / c2)
    roll = spread.rolling(120, min_periods=60)
    z = (spread - roll.mean()) / roll.std()
    sig = pd.DataFrame({"jjm_coke": (-np.sign(z) * (z.abs() > 1.0).astype(float)).fillna(0).values},
                        index=common)
    close = pd.DataFrame({"jjm_coke": c1.values}, index=common)
    return sig, close, d1["open"].reindex(common), 2.0, 0.10, 1


def build_zn():
    """ZN库存z52极端化 (LME锌库存, smoothed_zn)."""
    inv = CON2.execute(
        "SELECT trade_date, value FROM inventory_daily "
        "WHERE source='LME铜库存' AND value_col='锌-库存' ORDER BY trade_date").df()
    inv["trade_date"] = pd.to_datetime(inv["trade_date"])
    inv = inv.set_index("trade_date")["value"].sort_index()
    df = load_smoothed_full("zn")
    close = df["close"]
    common = inv.dropna().index.intersection(close.dropna().index)
    inv_c, close_c = inv[common], close[common]
    m = inv_c.rolling(52, min_periods=26).mean()
    sd = inv_c.rolling(52, min_periods=26).std().replace(0, np.nan)
    z = (inv_c - m) / sd
    sig = pd.Series(0.0, index=z.index)
    sig[z > 1.0] = 1.0
    sig[z < -1.0] = -1.0
    return sig.to_frame("ZN"), close_c.to_frame("ZN"), df["open"], 10.0, 0.08, 1


def strat_nav(sig, close, open_s, cost, tv, pit, ret_open_series=None):
    """单子策略 close/open 双版NAV (严格对齐口径).

    close: pnl[t] = pos[t-1] × (close[t]/close[t-1]-1)  信号T-2收盘→T-1收盘建仓
    open:  pnl[t] = pos[t-1] × (open[t]/open[t-1]-1)    信号T-2收盘→T-1开盘建仓
    pit: CU=0 (因子自带shift1); T10y/jjm/ZN=1 (信号不带shift)
    """
    from cta_shared.factor_research import vol_target_backtest
    res_c = vol_target_backtest(sig, close, cost_bps={sig.columns[0]: cost},
                                 target_vol=tv, pit_shift=pit)
    if ret_open_series is not None:
        ret_open = (ret_open_series.shift(-1) / ret_open_series - 1).reindex(close.index)
        res_o = vol_target_backtest(sig, close, cost_bps={sig.columns[0]: cost},
                                     target_vol=tv, pit_shift=pit,
                                     ret_df=ret_open.shift(1).to_frame(sig.columns[0]))
    else:
        res_o = res_c
    return res_c["nav"], res_o["nav"]


def combo_run(rets: pd.DataFrame, weights: dict, fac_A: pd.DataFrame,
               lev: pd.Series) -> tuple[pd.Series, pd.Series]:
    """净组合: 渐进A + 动态杠杆 + 换手成本."""
    wf = pd.DataFrame({col: weights.get(col, 0) * fac_A[col] for col in rets.columns})
    gross = levered_portfolio(rets * fac_A, weights, lev_series=lev,
                                cost_annual=DEFAULT_LEV_COST)
    tc = turnover_cost(wf, cost_bps=COST_BPS, w_main=W_MAIN, lev_series=lev)
    return gross, gross - tc


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    oos = r[r.index >= IS_END]
    return {"sharpe": ann / (r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            "ann_ret": ann, "max_dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0,
            "oos_shp": oos.mean() / oos.std() * np.sqrt(252) if len(oos) > 50 else 0}


def main():
    print("=" * 108)
    print("组合重构测试: 真实smoothed口径子策略 + open执行 + 渐进A + 动态杠杆 + 换手成本")
    print("=" * 108)

    # 1. 重建子策略 (真实口径)
    print("\n[1] 重建子策略NAV (smoothed真主连, close/open双版)")
    subs = {}
    for name, builder in [("CU溢价TC", build_cu), ("T10y", build_t10y),
                            ("jjm_coke", build_jjm), ("ZN库存", build_zn)]:
        sig, close, open_s, cost, tv, pit = builder()
        nav_c, nav_o = strat_nav(sig, close, open_s, cost, tv, pit, open_s)
        subs[name] = {"close": nav_c, "open": nav_o}
        st_c, st_o = stats(nav_c.pct_change(fill_method=None).fillna(0)), \
                      stats(nav_o.pct_change(fill_method=None).fillna(0))
        print(f"  {name:<10} close: Sharpe={st_c['sharpe']:+.3f} | "
              f"open: Sharpe={st_o['sharpe']:+.3f} (diff {st_o['sharpe']-st_c['sharpe']:+.3f})")
        nav_c.to_csv(REPORT_DIR / f"rebuild_subnav_{name}_close.csv", header=["nav"])
        nav_o.to_csv(REPORT_DIR / f"rebuild_subnav_{name}_open.csv", header=["nav"])

    # 2. 主CTA (保持close口径, 数据干净)
    main_nav = pd.read_csv(REPORT_DIR / "final_nav_lump.csv",
                            index_col=0, parse_dates=True).iloc[:, 0]
    main_ret = main_nav.pct_change(fill_method=None).fillna(0)

    # 3. 组合扫描
    print("\n[2] 组合权重方案对比 (open执行子策略)")
    print(f"{'方案':<52} {'净Shp':>7} {'净AR':>8} {'净DD':>8} {'净Cal':>7} {'OOS-Shp':>8}")
    print("-" * 100)

    port_defs = [
        ("R0: 主CTA+CU45%", {"CU溢价TC": 1.0}),
        ("R1: 主CTA+CU31.5%+T10y13.5% (替代IH50)", {"CU溢价TC": 0.7, "T10y": 0.3}),
        ("R2: 主CTA+CU25%+T10y10%+jjm10%",
          {"CU溢价TC": 0.5556, "T10y": 0.2222, "jjm_coke": 0.2222}),
        ("R3: 主CTA+CU22.5%+T10y11.25%+jjm11.25%",
          {"CU溢价TC": 0.5, "T10y": 0.25, "jjm_coke": 0.25}),
    ]
    rows = []
    for label, sub_w in port_defs:
        weights = {"主CTA": W_MAIN}
        for k, v in sub_w.items():
            weights[k] = v * (1 - W_MAIN)

        for exec_mode in ["open", "close"]:
            rets = pd.DataFrame({"主CTA": main_ret})
            for k in sub_w:
                rets[k] = subs[k][exec_mode].pct_change(fill_method=None).fillna(0)
            common = rets.dropna().index
            rets = rets.reindex(common).fillna(0)

            fac_A = multi_window_factors(rets, skip_cols=("主CTA",),
                                          windows=(20, 40, 60), cut=0.5)
            lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
            gross, net = combo_run(rets, weights, fac_A, lev)
            st = stats(net)
            mode_label = f"{label} [{exec_mode}]"
            print(f"{mode_label:<52} {st['sharpe']:>7.3f} {st['ann_ret']:>8.2%} "
                  f"{st['max_dd']:>8.2%} {st['calmar']:>7.2f} {st['oos_shp']:>8.3f}")
            rows.append({"方案": mode_label, **st})

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "rebuild_portfolio.csv", index=False)

    # 4. 最优方案年度收益 (open口径)
    print("\n[3] 最优方案 (open) 年度收益 vs 主CTA单独")
    best = df[df["方案"].str.contains("open")].sort_values("calmar", ascending=False).iloc[0]
    label_best = best["方案"].split(" [")[0]
    sub_w = dict(port_defs[[p[0] for p in port_defs].index(label_best)][1])
    weights = {"主CTA": W_MAIN}
    for k, v in sub_w.items():
        weights[k] = v * (1 - W_MAIN)
    rets = pd.DataFrame({"主CTA": main_ret})
    for k in sub_w:
        rets[k] = subs[k]["open"].pct_change(fill_method=None).fillna(0)
    common = rets.dropna().index
    rets = rets.reindex(common).fillna(0)
    fac_A = multi_window_factors(rets, skip_cols=("主CTA",), windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    _, net = combo_run(rets, weights, fac_A, lev)
    yr = net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr_main = main_ret.reindex(common).fillna(0).groupby(
        main_ret.reindex(common).index.year).apply(lambda x: (1 + x).prod() - 1)
    print(pd.DataFrame({"重构组合(净)": yr, "主CTA单独": yr_main}).round(4).to_string())
    (1 + net).cumprod().to_csv(REPORT_DIR / "rebuild_final_nav.csv", header=["nav"])
    print(f"\n★ 最优方案NAV留存: rebuild_final_nav.csv ({label_best})")


if __name__ == "__main__":
    main()
