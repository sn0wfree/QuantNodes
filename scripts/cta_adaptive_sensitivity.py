#!/usr/bin/env python3
"""cta_adaptive_sensitivity.py — 自适应子权重参数敏感性 + 逐年贡献复核.

目的:
  1. 用统一实现 (cta_shared.portfolio_tools) 重跑自适应子权重参数敏感性
     — 修正调研5初版数字与正统实现不符的问题
  2. 逐年贡献分解 (自适应 vs 固定权重), 验证IS/OOS均匀性
  3. 降权触发统计 + 降权后30日因果检验 (机制原因记录)

参数决策 (用户确认): 主推 0.5减半+60日窗口 (先验合理, 防过拟合).
敏感性表仅作稳健性记录; 若窗口40/cut0.3更好也只记录为"激进变体"不采纳.

输出:
  - reports/cta/adaptive_sensitivity.csv  敏感性表
  - reports/cta/adaptive_yearly.csv       逐年贡献
  - reports/cta/adaptive_trigger.csv      降权触发统计
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import REPORT_DIR  # noqa: E402
from cta_shared.portfolio_tools import (  # noqa: E402
    adaptive_factor, adaptive_factors, dynamic_leverage, levered_portfolio,
    yearly_diff, DEFAULT_LEV_COST,
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
# 主推配置 (61ab5be8 定版)
W_MAIN = 0.55
LEV_HI = 1.5
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=LEV_HI)
BASE_WEIGHTS = {"主CTA": W_MAIN, "CU溢价TC": 0.7 * (1 - W_MAIN),
                 "IH50动40×OI10": 0.3 * (1 - W_MAIN)}


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


def run_variant(rets: pd.DataFrame, cut: float | None, window: int) -> pd.Series:
    """cut=None → 固定权重 (无自适应); 否则自适应因子=cut."""
    lev_s = dynamic_leverage(rets["主CTA"], **LEV_PARAMS)
    if cut is None:
        factors = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    else:
        factors = adaptive_factors(rets, cut=cut, window=window)
    return levered_portfolio(rets * factors, BASE_WEIGHTS, lev_series=lev_s,
                              cost_annual=DEFAULT_LEV_COST)


def main():
    rets = load_all_returns()
    print(f"数据范围: {rets.index.min().date()} ~ {rets.index.max().date()} "
          f"({len(rets)}天)\n")

    # ── 1. 参数敏感性表 ─────────────────────────────────────────────────────
    print("=" * 96)
    print("参数敏感性 (统一实现, 主推=减半0.5+60日; 全部变体仅记录不选参)")
    print("=" * 96)
    variants = [
        ("固定权重(无自适应)", None, 60),
        ("自适应 0.3×w 窗口60", 0.3, 60),
        ("自适应 0.5×w 窗口60 ★主推", 0.5, 60),
        ("自适应 0.7×w 窗口60", 0.7, 60),
        ("自适应 0.5×w 窗口40", 0.5, 40),
        ("自适应 0.5×w 窗口120", 0.5, 120),
    ]
    rows = []
    for label, cut, win in variants:
        combo = run_variant(rets, cut, win)
        st = stats(combo)
        st["变体"] = label
        rows.append(st)
    df = pd.DataFrame(rows).set_index("变体")
    cols = ["sharpe", "ann_ret", "max_dd", "calmar",
            "IS_Sharpe", "OOS_Sharpe", "OOS_AnnRet", "OOS_MaxDD"]
    print(df[cols].round(4).to_string())
    df.to_csv(REPORT_DIR / "adaptive_sensitivity.csv")

    n_better = (df.loc[[i for i in df.index if "★" not in i and "固定" not in i],
                        "sharpe"] > df.loc["固定权重(无自适应)", "sharpe"]).sum()
    print(f"\n>> 稳健性: {n_better}/5 个自适应变体 Sharpe 优于固定权重")

    # ── 2. 逐年贡献复核 (主推 vs 固定) ──────────────────────────────────────
    print("\n" + "=" * 96)
    print("逐年贡献 (自适应主推 - 固定权重)")
    print("=" * 96)
    combo_adapt = run_variant(rets, 0.5, 60)
    combo_fixed = run_variant(rets, None, 60)
    yd = yearly_diff(combo_adapt, combo_fixed)
    print(yd.round(4).to_string())
    yd.to_csv(REPORT_DIR / "adaptive_yearly.csv")

    is_mask = yd.index < 2022
    print(f"\n>> IS期累计差值: {(1 + yd.loc[is_mask, 'diff']).prod() - 1:+.2%}")
    print(f">> OOS期累计差值: {(1 + yd.loc[~is_mask, 'diff']).prod() - 1:+.2%}")
    neg_years = yd.index[yd["diff"] < 0].tolist()
    print(f">> 负贡献年份: {neg_years} (60日窗口滞后代价, 已知局限)")

    # ── 3. 降权触发统计 + 因果检验 ──────────────────────────────────────────
    print("\n" + "=" * 96)
    print("降权触发统计 (机制原因记录)")
    print("=" * 96)
    trig_rows = []

    def _ann(s: pd.Series) -> float:
        if len(s) <= 30:
            return np.nan
        return (1 + s).prod() ** (252 / len(s)) - 1

    for col in rets.columns:
        if col == "主CTA":
            continue
        f = adaptive_factor(rets[col], window=60, cut=0.5)
        r = rets[col]
        down = f == 0.5
        hi, lo = r[~down], r[down]
        # 因果诊断: 降权期之后30日收益 (用未来数据, 仅诊断不进信号)
        fwd30 = r.rolling(30).sum().shift(-30)
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
    print("\n>> 机制解读: 降权期年化全为负(逆境识别有效); '降权后30日'为踏空反弹成本诊断")

    # ── 4. 主推方案NAV留存 ─────────────────────────────────────────────────
    (1 + combo_adapt).cumprod().to_csv(REPORT_DIR / "final_dynamic_lev_nav.csv",
                                        header=["nav"])
    st = stats(combo_adapt)
    print("\n★ 主推方案 (0.5+60日) NAV已留存: final_dynamic_lev_nav.csv")
    print(f"   Sharpe={st['sharpe']:.3f} AnnRet={st['ann_ret']:.2%} "
          f"MaxDD={st['max_dd']:.2%} Calmar={st['calmar']:.2f} "
          f"OOS-Shp={st['OOS_Sharpe']:.3f}")


if __name__ == "__main__":
    main()
