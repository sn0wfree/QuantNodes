#!/usr/bin/env python3
"""cta_adaptive_sensitivity.py — 自适应子权重参数敏感性 + 渐进降权 + 换手成本.

目的:
  1. 参数敏感性表 (统一实现, 保守参数 0.5+60日为主推基线)
  2. 渐进降权方案 (多窗口20/40/60独立相乘) — 你的设计
  3. 显式换手成本量化 — 权重调制是真实仓位调整
  4. 逐年贡献分解 (验证IS/OOS均匀性)
  5. 降权触发统计 + 降权后30日因果检验

参数决策:
  主推 = 渐进A (20×40×60相乘, cut=0.5) — 净Sharpe 3.067 / Calmar 4.67 / OOS 2.578.
  单窗口20日作为备选 (机制更简单但净口径三项中两项略输).

输出:
  - reports/cta/adaptive_sensitivity.csv       完整敏感性表 (含换手成本)
  - reports/cta/adaptive_gradual_levels.csv   渐进A档位分布
  - reports/cta/adaptive_yearly.csv           逐年贡献 (净口径)
  - reports/cta/adaptive_trigger.csv          降权触发统计
  - reports/cta/adaptive_cost_sensitivity.csv 换手费率敏感性
  - reports/cta/final_dynamic_lev_nav.csv     主推方案净NAV
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import REPORT_DIR  # noqa: E402
from cta_shared.portfolio_tools import (  # noqa: E402
    adaptive_factors, multi_window_factors,
    dynamic_leverage, levered_portfolio, turnover_cost, yearly_diff,
    DEFAULT_LEV_COST,
)

warnings.filterwarnings("ignore")

IS_END = "2022-01-01"
NAV_FILES = {
    "主CTA": "final_nav_lump.csv",
    "ZN库存": "zn_inventory_nav.csv",
    "CU溢价TC": "cu_premium_nav.csv",
    "IH50动40×OI10": "ih50_oi_nav.csv",
    "T10y OI变化5": "t10y_oi_nav.csv",
    "rbi_steel": "chain_rbi_steel_nav.csv",
    "jjm_coke": "chain_jjm_coke_nav.csv",
    "ym_oilmeal": "chain_ym_oilmeal_nav.csv",
}
W_MAIN = 0.55
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
BASE_WEIGHTS = {"主CTA": W_MAIN, "CU溢价TC": 0.7 * (1 - W_MAIN),
                 "IH50动40×OI10": 0.3 * (1 - W_MAIN)}
COST_BPS_DEFAULT = 5.0


def load_all_returns() -> pd.DataFrame:
    rets, common = {}, None
    for name, fn in NAV_FILES.items():
        nav = pd.read_csv(REPORT_DIR / fn, index_col=0, parse_dates=True).iloc[:, 0]
        r = nav.pct_change(fill_method=None).fillna(0)
        rets[name] = r
        common = r.index if common is None else common.intersection(r.index)
    return pd.DataFrame({k: v.reindex(common) for k, v in rets.items()})


def stats(r: pd.Series, is_end: str = IS_END) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()

    def _one(s):
        if len(s) < 50:
            return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}
        nav_r = (1 + s).cumprod()
        a = nav_r.iloc[-1] ** (252 / len(s)) - 1
        v = s.std() * np.sqrt(252)
        d = (nav_r / nav_r.cummax() - 1).min()
        return {"sharpe": a / v if v > 0 else 0, "ann_ret": a, "max_dd": d}

    oos = r[r.index >= is_end]
    return {
        "sharpe": ann / vol if vol > 0 else 0, "ann_ret": ann,
        "max_dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0,
        "IS_Sharpe": _one(r[r.index < is_end])["sharpe"],
        "OOS_Sharpe": _one(oos)["sharpe"],
        "OOS_AnnRet": _one(oos)["ann_ret"],
        "OOS_MaxDD": _one(oos)["max_dd"],
    }


def panel_fixed() -> pd.DataFrame:
    rets = globals().get("_rets_cache")  # set in main()
    return pd.DataFrame(1.0, index=rets.index, columns=rets.columns)


def net_combo(rets: pd.DataFrame, factors: pd.DataFrame, cost_bps: float = COST_BPS_DEFAULT):
    """无成本组合 / 含换手成本组合 / 换手总量, 三件套返回."""
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    gross = levered_portfolio(rets * factors, BASE_WEIGHTS, lev_series=lev,
                                cost_annual=DEFAULT_LEV_COST)
    wf = pd.DataFrame({col: BASE_WEIGHTS.get(col, 0) * factors[col] for col in rets.columns})
    tc = turnover_cost(wf, cost_bps=cost_bps, w_main=W_MAIN, lev_series=lev)
    net = gross - tc
    return gross, net, tc, lev


def main():
    rets = load_all_returns()
    global _rets_cache
    _rets_cache = rets
    print(f"数据范围: {rets.index.min().date()} ~ {rets.index.max().date()} "
          f"({len(rets)}天)\n")

    # ── 1. 参数敏感性表 (净口径, 5bp换手成本) ────────────────────────────
    print("=" * 100)
    print("1. 参数敏感性 (净口径: 含5bp换手成本, scatter=W×f 切换+杠杆调整)")
    print("=" * 100)
    variants = [
        ("固定权重", panel_fixed()),
        ("自适应 0.5×w 窗口60", adaptive_factors(rets, cut=0.5, window=60)),
        ("自适应 0.5×w 窗口40", adaptive_factors(rets, cut=0.5, window=40)),
        ("自适应 0.5×w 窗口20", adaptive_factors(rets, cut=0.5, window=20)),
        ("自适应 0.3×w 窗口60", adaptive_factors(rets, cut=0.3, window=60)),
        ("渐进A 20×40×60 ★主推", multi_window_factors(rets, windows=(20, 40, 60), cut=0.5)),
        ("渐进A 20×40×60 cut0.4", multi_window_factors(rets, windows=(20, 40, 60), cut=0.4)),
    ]
    rows = []
    for label, fac in variants:
        gross, net, tc, lev = net_combo(rets, fac)
        st_g, st_n = stats(gross), stats(net)
        wf = pd.DataFrame({col: BASE_WEIGHTS.get(col, 0) * fac[col] for col in rets.columns})
        annual_turn = wf.diff().abs().sum(axis=1).mean() * 252
        sub_cost = tc.mean() * 252
        rows.append({"变体": label, "换手/年": annual_turn,
                      "切换成本/年": sub_cost,
                      "净Sharpe": st_n["sharpe"], "净AnnRet": st_n["ann_ret"],
                      "净MaxDD": st_n["max_dd"], "净Calmar": st_n["calmar"],
                      "OOS_Sharpe": st_n["OOS_Sharpe"], "OOS_AnnRet": st_n["OOS_AnnRet"],
                      "无成本Sharpe": st_g["sharpe"]})
    df = pd.DataFrame(rows).set_index("变体")
    cols = ["换手/年", "切换成本/年", "净Sharpe", "净AnnRet", "净MaxDD",
            "净Calmar", "OOS_Sharpe", "OOS_AnnRet", "无成本Sharpe"]
    print(df[cols].round(4).to_string())
    df.to_csv(REPORT_DIR / "adaptive_sensitivity.csv")

    # ── 2. 渐进A档位分布 ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("2. 渐进A 档位分布 (1.0/0.5/0.25/0.125)")
    print("=" * 100)
    fac_A = multi_window_factors(rets, windows=(20, 40, 60), cut=0.5)
    level_rows = []
    for col in fac_A.columns:
        if col == "主CTA":
            continue
        f = fac_A[col]
        row = {"子策略": col,
                "1.0": (f == 1.0).mean(),
                "0.5": (f == 0.5).mean(),
                "0.25": (f == 0.25).mean(),
                "0.125": (f == 0.125).mean()}
        level_rows.append(row)
    levels = pd.DataFrame(level_rows).set_index("子策略")
    print(levels.round(3).to_string())
    levels.to_csv(REPORT_DIR / "adaptive_gradual_levels.csv")

    # ── 3. 逐年贡献 (净口径) ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("3. 逐年贡献 (主推渐进A净 vs 单窗口20日净 vs 固定)")
    print("=" * 100)
    gross_20, net_20, _, _ = net_combo(rets, adaptive_factors(rets, cut=0.5, window=20))
    gross_A, net_A, _, _ = net_combo(rets, fac_A)
    yd = yearly_diff(net_A, net_20)
    yd.columns = ["渐进A净", "单20日净", "diff(A-20)"]
    print(yd.round(4).to_string())
    yd.to_csv(REPORT_DIR / "adaptive_yearly.csv")
    im = yd.index < 2022
    print(f"\n>> IS累计差值 (A-20日): {(1 + yd.loc[im, 'diff(A-20)']).prod() - 1:+.2%}")
    print(f">> OOS累计差值 (A-20日): {(1 + yd.loc[~im, 'diff(A-20)']).prod() - 1:+.2%}")
    neg = yd.index[yd["diff(A-20)"] < 0].tolist()
    print(f">> 负贡献年份: {neg}")

    # ── 4. 降权触发统计 ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("4. 降权触发统计 + 因果检验 (主推方案)")
    print("=" * 100)
    trig_rows = []

    def _ann(s):
        if len(s) <= 30:
            return np.nan
        return (1 + s).prod() ** (252 / len(s)) - 1

    for col in rets.columns:
        if col == "主CTA":
            continue
        # 渐进A的触发判定: 任一窗口触发即视为降权(0.5或更深)
        f = fac_A[col]
        down = f < 1.0
        hi, lo = rets[col][~down], rets[col][down]
        fwd30 = rets[col].rolling(30).sum().shift(-30)
        fwd_dn = fwd30[down].dropna()
        fwd_up = fwd30[~down].dropna()
        trig_rows.append({
            "子策略": col,
            "降权占比": down.mean(),
            "降权期年化": _ann(lo),
            "正常期年化": _ann(hi),
            "降权后30日年化": fwd_dn.mean() * 252 / 30 if len(fwd_dn) > 10 else np.nan,
            "正常期后30日年化": fwd_up.mean() * 252 / 30 if len(fwd_up) > 10 else np.nan,
        })
    trig = pd.DataFrame(trig_rows).set_index("子策略")
    print(trig.round(4).to_string())
    trig.to_csv(REPORT_DIR / "adaptive_trigger.csv")

    # ── 5. 换手费率敏感性 (主推方案) ────────────────────────────────────
    print("\n" + "=" * 100)
    print("5. 主推方案换手费率敏感性 (假设各子策略/主CTA统一费率)")
    print("=" * 100)
    sens_rows = []
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    for bps in [1, 3, 5, 10, 15, 20]:
        wf = pd.DataFrame({col: BASE_WEIGHTS.get(col, 0) * fac_A[col] for col in rets.columns})
        gross = levered_portfolio(rets * fac_A, BASE_WEIGHTS, lev_series=lev,
                                    cost_annual=DEFAULT_LEV_COST)
        tc = turnover_cost(wf, cost_bps=bps, w_main=W_MAIN, lev_series=lev)
        net = gross - tc
        st = stats(net)
        annual_turn = wf.diff().abs().sum(axis=1).mean() * 252
        sens_rows.append({"费率bp": bps, "换手/年": annual_turn,
                          "净Sharpe": st["sharpe"], "净AnnRet": st["ann_ret"],
                          "净MaxDD": st["max_dd"], "净Calmar": st["calmar"],
                          "OOS_Sharpe": st["OOS_Sharpe"]})
    sens = pd.DataFrame(sens_rows).set_index("费率bp")
    print(sens.round(4).to_string())
    sens.to_csv(REPORT_DIR / "adaptive_cost_sensitivity.csv")

    # ── 6. 主推方案净NAV留存 ─────────────────────────────────────────────
    (1 + net_A).cumprod().to_csv(REPORT_DIR / "final_dynamic_lev_nav.csv",
                                    header=["nav"])
    st = stats(net_A)
    print("\n★ 主推 (渐进A净) NAV留存: final_dynamic_lev_nav.csv")
    print(f"   净Sharpe={st['sharpe']:.3f} 净AnnRet={st['ann_ret']:.2%} "
          f"净MaxDD={st['max_dd']:.2%} 净Calmar={st['calmar']:.2f} "
          f"OOS-Shp={st['OOS_Sharpe']:.3f}")


if __name__ == "__main__":
    main()
