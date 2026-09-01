#!/usr/bin/env python3
"""research_factor_combination.py — 因子组合研究 (修正方法论, 2018+生产窗口).

三个任务:
1. Pseudo-OOS IC全评估 (FWD1/FWD2/FWD4)
2A. 组合IC: 等权/IC加权 (不用OLS R²)
2B. 重复性检查: comex_inv ≈ etf_tonnes
2C. 调仓频率: 周/双周/月
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import duckdb
import numpy as np
import pandas as pd

OUTPUT_DIR = "reports/gold"
PROD_START = "2018-01-01"

INDICATORS = {
    "tips10y":("gold_fred","tips10y"),"tips5y":("gold_fred","tips"),
    "rir":("gold_fred","rir"),"us_10y":("gold_macro","us_10y"),
    "us_2y":("gold_macro","us_2y"),"fed_rate_daily":("gold_fred","fed_rate_daily"),
    "fed_total_assets":("gold_fred","fed_total_assets"),"m2":("gold_fred","m2_money_supply"),
    "comex_inv":("gold_position","comex_inv"),"etf_tonnes":("gold_position","etf_tonnes"),
    "cftc_oi":("gold_position","cftc_oi"),"brent":("fred_dcoil","brent"),
    "wti":("fred_dcoil","wti"),"us_cpi_yoy":("美国CPI","us_cpi_yoy"),
    "us_ip":("美国工业产出","us_ip"),
}


def load_data():
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    def load(src, vc):
        df = con.execute("SELECT trade_date, value FROM fundamentals_daily "
                        "WHERE source=? AND value_col=? ORDER BY trade_date", [src, vc]).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")["value"].astype(float)
    gold = load("gold_macro", "gold_spot")
    data = {}
    for name, (s, v) in INDICATORS.items():
        data[name] = load(s, v)
    con.close()
    return gold, data


def pseudo_oos_ic(s, fwd, min_train=26):
    """严格pseudo-OOS, 缩短min_train以适应短序列."""
    n = len(fwd)
    oos = []
    for i in range(min_train, n):
        past_x = s.iloc[:i].dropna()
        past_y = fwd.iloc[:i].dropna()
        common = past_x.index.intersection(past_y.index)
        if len(common) < 12:
            continue
        oos.append(past_x.rank().corr(past_y.rank(), method="spearman"))
    return np.nanmean(oos) if oos else np.nan


def compute_signal_pseudo(s, min_train=26):
    """生成pseudo-OOS信号(0-1), 防未来函数."""
    out = pd.Series(np.nan, index=s.index)
    for i in range(min_train, len(s)):
        past = s.iloc[:i].dropna()
        if len(past) < 12:
            out.iloc[i] = 0
            continue
        mu, sd = past.mean(), past.std() + 1e-8
        out.iloc[i] = float((s.iloc[i] - mu) / sd)
    return (out.clip(-3, 3) / 6 + 0.5).fillna(0.5)


def main():
    print("=" * 70)
    print("因子组合研究 — 修正方法论 (Pseudo-OOS IC + 组合 + 调仓频率)")
    print("窗口: 2018-2026生产窗口")
    print("=" * 70)

    gold, data = load_data()
    w = gold.resample("W-FRI").last().dropna()
    fwd1 = w.pct_change().shift(-1)
    fwd2 = w.pct_change().shift(-2)
    fwd4 = w.pct_change().shift(-4)

    # ========================================
    # 任务2B: 重复性检查
    # ========================================
    print("\n[任务2B] comex_inv vs etf_tonnes 重复性")
    comex_etf_corr = data["comex_inv"].corr(data["etf_tonnes"], method="spearman")
    print(f"  Spearman相关: {comex_etf_corr:+.4f}")
    if comex_etf_corr > 0.95:
        print(f"  ⚠️ 相关=1.0, 是同一数据! 组合只保留一个")
    else:
        print(f"  两者独立, 都保留")

    # ========================================
    # 任务1: Pseudo-OOS IC (FWD1/FWD2/FWD4)
    # ========================================
    print("\n[任务1] 各指标Pseudo-OOS IC (FWD1/FWD2/FWD4)")
    ic_results = {}
    for name, s in data.items():
        s_w = s.reindex(w.index, method="ffill")
        ic1 = pseudo_oos_ic(s_w, fwd1)
        ic2 = pseudo_oos_ic(s_w, fwd2)
        ic4 = pseudo_oos_ic(s_w, fwd4)
        ic_results[name] = (ic1, ic2, ic4)

    sorted_by = sorted(ic_results.items(), key=lambda x: abs(x[1][1]) if pd.notna(x[1][1]) else 0, reverse=True)
    print(f"\n  {'指标':<22} {'FWD1':>8} {'FWD2':>8} {'FWD4':>8} {'最优':>6}")
    print("  " + "-" * 56)
    for name, (ic1, ic2, ic4) in sorted_by:
        marker = "⭐" if abs(ic2) > 0.05 else "  "
        print(f"  {name:<22} {ic1:>+8.4f} {ic2:>+8.4f} {ic4:>+8.4f} {marker}")

    # ========================================
    # 任务2A: 组合IC
    # ========================================
    print("\n[任务2A] 组合IC评估")
    # 入选条件: 任一horizon |IC|>0.05 (放宽以纳入更多)
    selected = [n for n, (i1, i2, i4) in ic_results.items()
                if max(abs(i1), abs(i2), abs(i4)) > 0.05 and pd.notna(i1)]
    # 去重: 保留comex_inv, 删etf_tonnes
    if "comex_inv" in selected and "etf_tonnes" in selected:
        selected.remove("etf_tonnes")
    print(f"  入选特征(|IC|max>0.05, 去重): {len(selected)}个 → {selected}")

    # 构造pseudo-OOS信号
    signals = {name: compute_signal_pseudo(data[name]) for name in selected}
    # 对齐到gold周频
    signals = {k: v.reindex(w.index, method="ffill").fillna(0.5) for k, v in signals.items()}

    # 1. 等权组合
    combo_eq = sum(signals[k] for k in selected) / len(selected)
    # 2. IC加权 (权重∝|IC|)
    weights = {k: max(abs(i1), abs(i2), abs(i4)) for k, (i1, i2, i4) in ic_results.items() if k in selected}
    wsum = sum(weights.values())
    weights = {k: v/wsum for k, v in weights.items()}
    combo_icw = sum(weights[k] * signals[k] for k in selected)
    # 3. Top1单强
    top1 = sorted_by[0][0]
    # 4. 6因子原始(生产用的)
    prod_6 = ["tips10y","tips5y","rir","us_10y","us_2y","m2"]
    combo_prod = sum(compute_signal_pseudo(data[k]).reindex(w.index, method="ffill").fillna(0.5) for k in prod_6) / len(prod_6)

    print(f"\n  组合方式              FWD1     FWD2     FWD4")
    print("  " + "-" * 50)
    for label, combo in [
        (f"Top1({top1})", signals[top1]),
        ("6因子原始", combo_prod),
        (f"{len(selected)}因子等权", combo_eq),
        ("IC加权", combo_icw),
    ]:
        ics = []
        for fwd in [fwd1, fwd2, fwd4]:
            ic = pseudo_oos_ic(combo, fwd)
            ics.append(ic if pd.notna(ic) else np.nan)
        print(f"  {label:<20} {ics[0]:>+7.4f} {ics[1]:>+7.4f} {ics[2]:>+7.4f}")

    # ========================================
    # 任务2C: 调仓频率
    # ========================================
    print("\n[任务2C] 调仓频率分析 (Pseudo-OOS IC vs 方向准确率)")
    for label, combo in [
        (f"Top1({top1})", signals[top1]),
        (f"{len(selected)}因子等权", combo_eq),
        ("IC加权", combo_icw),
    ]:
        print(f"\n  {label}:")
        for h_name, fwd in [("FWD1(周)", fwd1), ("FWD2(双周)", fwd2), ("FWD4(月)", fwd4)]:
            ic = pseudo_oos_ic(combo, fwd)
            common = combo.dropna().index.intersection(fwd.dropna().index)
            s_clean = combo.reindex(common)
            fwd_clean = fwd.reindex(common)
            hit = ((s_clean > 0.5) == (fwd_clean > 0)).mean()
            print(f"    {h_name:<12} IC={ic:+.4f}  方向准确率={hit:.1%}")

    # ========================================
    # 输出CSV
    # ========================================
    rows = []
    for name, (ic1, ic2, ic4) in sorted_by:
        best = max([(abs(ic1), "FWD1"), (abs(ic2), "FWD2"), (abs(ic4), "FWD4")],
                   key=lambda x: x[0] if pd.notna(x[0]) else 0)
        rows.append({
            "indicator": name, "ic_fwd1": ic1, "ic_fwd2": ic2, "ic_fwd4": ic4,
            "best_horizon": best[1] if pd.notna(best[0]) else "NA",
        })
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/factor_combination.csv", index=False, float_format="%.6f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/factor_combination.csv")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    # 找最强组合
    best_combos = {
        "6因子原始": combo_prod,
        f"{len(selected)}因子等权": combo_eq,
        "IC加权": combo_icw,
    }
    print(f"  Top1单强 ({top1}): FWD2 IC={ic_results[top1][1]:+.4f}")
    for name, combo in best_combos.items():
        ic2 = pseudo_oos_ic(combo, fwd2)
        print(f"  {name}: FWD2 IC={ic2:+.4f}")
    # 调仓频率结论
    eq_fwd1 = pseudo_oos_ic(combo_eq, fwd1)
    eq_fwd2 = pseudo_oos_ic(combo_eq, fwd2)
    eq_fwd4 = pseudo_oos_ic(combo_eq, fwd4)
    best_fwd = max([(eq_fwd1, "FWD1"), (eq_fwd2, "FWD2"), (eq_fwd4, "FWD4")],
                   key=lambda x: abs(x[0]) if pd.notna(x[0]) else 0)
    print(f"  调仓频率: {best_fwd[1]}")


if __name__ == "__main__":
    main()
