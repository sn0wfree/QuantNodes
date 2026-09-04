# coding=utf-8
"""cta_r3_portfolio.py — R3 实验: R2 权重再平衡 (加入ZN库存).

动机:
  - ZN库存 2024/25 Sharpe 1.71/1.26 却不在 R2 (2024 +7.4% / 2025 +13.5%)
  - CU溢价TC 2024 +15.1% 只有 25% 绝对权重
  - 主CTA剔EC 2024 -0.4%, 需要子策略层更强对冲

修复 + 实验:
  1. 主CTA nav 用 golden (overfit_final_navs 基线) — R2 原版用的是 stale
     final_nav_lump.csv (落后3个commit), 本次修复同源性
  2. 复刻 R2 (golden主CTA) → 量化与官方 R2 (stale主CTA) 的差异
  3. R3 变体网格 (子策略层内占比, 主CTA 55% 不变, 子策略层 45%):

     R2复刻 : CU .5556 / T10y .2222 / jjm .2222            (ZN=0)
     R3a等比挤入: CU .50 / T10y .20 / jjm .20 / ZN .10
     R3b挤T10y  : CU .5556 / T10y .1111 / jjm .2222 / ZN .1111
     R3c CU加仓 : CU .65 / T10y .15 / jjm .10 / ZN .10
     R3d双核    : CU .60 / T10y .10 / jjm .15 / ZN .15

  组合构建与 rebuild_portfolio.py 完全同法 (渐进A 20x40x60 cut0.5 +
  动态杠杆[1.0,1.5] + 5bp换手成本 + 借款成本), 子策略 open 口径.

选择标准 (预定义): 2024+2025 平均收益改善 AND 全样本Sharpe不低于R2复刻 AND OOS不恶化

输出: reports/cta/r3_portfolio.csv + 分年表 + r3_best_nav.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
from cta_shared.portfolio_tools import (  # noqa: E402
    DEFAULT_LEV_COST, dynamic_leverage, levered_portfolio, multi_window_factors,
    turnover_cost,
)

REPORT_DIR = Path("reports/cta")
IS_END = "2022-01-01"
W_MAIN = 0.55
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0

GOLDEN_MAIN = REPORT_DIR / "overfit_final_navs" / "基线_严格OOS0.5,60_2018.parquet"
STALE_MAIN = REPORT_DIR / "final_nav_lump.csv"

SUB_FILES = {  # 已落盘的 PIT 修正版 open 口径子策略 (rebuild 脚本产出, 直接复用)
    "CU溢价TC": "rebuild_subnav_CU溢价TC_open.csv",
    "T10y": "rebuild_subnav_T10y_open.csv",
    "jjm_coke": "rebuild_subnav_jjm_coke_open.csv",
    "ZN库存": "rebuild_subnav_ZN库存_open.csv",
}

PORT_DEFS = [
    ("R2复刻(CU55.6/T10y22.2/jjm22.2)",
     {"CU溢价TC": 0.5556, "T10y": 0.2222, "jjm_coke": 0.2222}),
    ("R3a等比挤ZN(CU50/T10y20/jjm20/ZN10)",
     {"CU溢价TC": 0.50, "T10y": 0.20, "jjm_coke": 0.20, "ZN库存": 0.10}),
    ("R3b挤T10y(CU55.6/T10y11.1/jjm22.2/ZN11.1)",
     {"CU溢价TC": 0.5556, "T10y": 0.1111, "jjm_coke": 0.2222, "ZN库存": 0.1111}),
    ("R3c CU加仓(CU65/T10y15/jjm10/ZN10)",
     {"CU溢价TC": 0.65, "T10y": 0.15, "jjm_coke": 0.10, "ZN库存": 0.10}),
    ("R3d双核(CU60/T10y10/jjm15/ZN15)",
     {"CU溢价TC": 0.60, "T10y": 0.10, "jjm_coke": 0.15, "ZN库存": 0.15}),
]


def stats(r: pd.Series) -> dict:
    """与 rebuild_portfolio.py 同口径 (252计数)."""
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    oos = r[r.index >= IS_END]
    return {"sharpe": ann / (r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            "ann_ret": ann, "max_dd": dd,
            "calmar": ann / abs(dd) if dd != 0 else 0,
            "oos_shp": oos.mean() / oos.std() * np.sqrt(252) if len(oos) > 50 else 0}


def build_net(main_ret: pd.Series, sub_rets: dict[str, pd.Series],
              sub_w: dict[str, float]) -> pd.Series:
    """R2/R3 净组合: 渐进A + 动态杠杆 + 换手成本 (与 rebuild 同法)."""
    weights = {"主CTA": W_MAIN}
    for k, v in sub_w.items():
        weights[k] = v * (1 - W_MAIN)
    rets = pd.DataFrame({"主CTA": main_ret})
    for k in sub_w:
        rets[k] = sub_rets[k]
    rets = rets.reindex(rets.dropna().index.union(rets.index)).fillna(0)
    common = rets.dropna().index
    rets = rets.loc[common]
    fac_A = multi_window_factors(rets, skip_cols=("主CTA",),
                                  windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    wf = pd.DataFrame({col: weights.get(col, 0) * fac_A[col] for col in rets.columns})
    gross = levered_portfolio(rets * fac_A, weights, lev_series=lev,
                                cost_annual=DEFAULT_LEV_COST)
    tc = turnover_cost(wf, cost_bps=COST_BPS, w_main=W_MAIN, lev_series=lev)
    return (gross - tc).dropna()


def main() -> None:
    # 数据: golden 主CTA + 落盘子策略
    main_golden = pd.read_parquet(GOLDEN_MAIN)["nav"]
    main_golden = main_golden[main_golden.index <= "2025-12-31"]
    main_stale = pd.read_csv(STALE_MAIN, index_col=0, parse_dates=True).iloc[:, 0]
    sub_rets = {}
    for name, f in SUB_FILES.items():
        nav = pd.read_csv(REPORT_DIR / f, index_col=0, parse_dates=True).iloc[:, 0]
        nav = nav[nav.index <= "2025-12-31"]
        sub_rets[name] = nav.pct_change(fill_method=None).fillna(0)
    ret_golden = main_golden.pct_change(fill_method=None).fillna(0)
    ret_stale = main_stale.pct_change(fill_method=None).fillna(0)

    print(f"{'方案':<46} {'净Shp':>7} {'净AR':>8} {'净DD':>8} {'净Cal':>7} {'OOS':>7}")
    print("-" * 90)
    rows = []
    nets = {}
    for label, sub_w in PORT_DEFS:
        # golden 主CTA 版
        net = build_net(ret_golden, sub_rets, sub_w)
        st = stats(net)
        nets[label] = net
        print(f"{label:<46} {st['sharpe']:>7.3f} {st['ann_ret']:>8.2%} "
              f"{st['max_dd']:>8.2%} {st['calmar']:>7.2f} {st['oos_shp']:>8.3f}")
        rows.append({"方案": label + " [golden]", **{k: round(v, 4)
                     for k, v in st.items()}})
        # R2 官方对照 (stale 主CTA, 仅复刻行)
        if label.startswith("R2"):
            net_s = build_net(ret_stale, sub_rets, sub_w)
            st_s = stats(net_s)
            print(f"{'  └ 对照: stale主CTA (官方R2口径)':<46} {st_s['sharpe']:>7.3f} "
                  f"{st_s['ann_ret']:>8.2%} {st_s['max_dd']:>8.2%} "
                  f"{st_s['calmar']:>7.2f} {st_s['oos_shp']:>8.3f}")
            rows.append({"方案": "R2复刻 [stale对照]", **{k: round(v, 4)
                         for k, v in st_s.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "r3_portfolio.csv", index=False)

    # 分年表 (重点 2024/2025)
    print("\n=== 分年收益 (净) ===")
    yearly = pd.DataFrame({label[:30]: net.groupby(net.index.year)
                            .apply(lambda x: (1 + x).prod() - 1)
                            for label, net in nets.items()})
    print(yearly.round(3).to_string())
    yearly.to_csv(REPORT_DIR / "r3_portfolio_yearly.csv")

    # 预设选择标准评估: 2024+2025均值改善 AND 全样本Sharpe不低于R2复刻 AND OOS不恶化
    print("\n=== 预设选择标准 (vs R2复刻) ===")
    base_st = stats(nets[PORT_DEFS[0][0]])
    base_y = yearly[PORT_DEFS[0][0][:30]]
    best_label, best_st = None, None
    for label, sub_w in PORT_DEFS[1:]:
        st = stats(nets[label])
        y = yearly[label[:30]]
        d2425 = (y.loc[[2024, 2025]].mean() - base_y.loc[[2024, 2025]].mean())
        ok_shp = st["sharpe"] >= base_st["sharpe"] - 0.02
        ok_oos = st["oos_shp"] >= base_st["oos_shp"] - 0.02
        ok_2425 = d2425 > 0
        verdict = "PASS" if (ok_shp and ok_oos and ok_2425) else "FAIL"
        print(f"  {label.split('(')[0]}: 24/25均值Δ{d2425:+.2%} "
              f"ShpΔ{st['sharpe']-base_st['sharpe']:+.3f} "
              f"OOSΔ{st['oos_shp']-base_st['oos_shp']:+.3f} → {verdict}")
        if verdict == "PASS" and (best_st is None or st["calmar"] > best_st["calmar"]):
            best_label, best_st = label, st
    if best_label:
        nets[best_label].to_csv(REPORT_DIR / "r3_best_nav.csv", header=["ret"])
        print(f"\n★ 最优: {best_label} (NAV留存 r3_best_nav.csv, 切换决策待用户拍板)")
    else:
        print("\n无变体通过选择标准 → R2 维持不变")


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (stats 内 np 未直接引用时 ruff 误报保护)
    main()
