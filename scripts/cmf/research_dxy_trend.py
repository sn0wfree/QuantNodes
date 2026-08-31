#!/usr/bin/env python3
"""scripts/cmf/research_dxy_trend.py — DXY变化趋势与黄金关系研究.

纯只读脚本, 不修改DuckDB.
检验7个DXY变化趋势变体的独立预测力, 与F10_dxy_z(水平)对比.

预注册门槛:
    G1: 偏IC(20d_vol) ≥ +0.08
    G2: ≥2/3子周期偏IC > 0
    G3: IC(5d_vol) ≥ +0.05
    G4: 与F10_dxy_z相关系数 < 0.7
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = "data/cmf/fundamental.duckdb"

SUBPERIODS = [
    ("2008-2013", "2008-01-01", "2013-12-31"),
    ("2014-2019", "2014-01-01", "2019-12-31"),
    ("2020-2025", "2020-01-01", "2025-12-31"),
]

GATES = {
    "G1_pIC_20d_vol_min": 0.08,
    "G2_subperiod_min": 2 / 3,
    "G3_IC_5d_vol_min": 0.05,
    "G4_f10_corr_max": 0.7,
}


def load_data() -> pd.DataFrame:
    """加载DXY及相关指标."""
    con = duckdb.connect(DB_PATH, read_only=True)
    queries_ = {
        "gold": ("gold_macro", "gold_spot"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "dxy": ("gold_macro", "dxy"),
        "epu": ("gold_fred", "epu"),
    }
    dfs = {}
    for col, (src, vc) in queries_.items():
        try:
            df = con.execute(
                f"SELECT trade_date, value FROM fundamentals_daily "
                f"WHERE source='{src}' AND value_col='{vc}' ORDER BY trade_date"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            dfs[col] = df.set_index("trade_date")["value"]
        except Exception as e:
            print(f"  ⚠️ {col} 加载失败: {e}")
            dfs[col] = pd.Series(dtype=float)
    con.close()
    return pd.DataFrame(dfs).sort_index().ffill().dropna()


def _z(s: pd.Series, window: int = 156) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """构建7个DXY变化趋势变体 + 基准F10."""
    out = df.copy()
    out["gold_ret"] = out["gold"].pct_change()
    for k in [5, 20]:
        out[f"fut_vol_{k}"] = (
            out["gold_ret"].shift(-1).rolling(k).std().shift(-(k - 1)) * np.sqrt(252)
        )
        out[f"fut_ret_{k}"] = out["gold"].shift(-k) / out["gold"] - 1
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)

    # F10基准 (DXY水平 zscore)
    out["F10_dxy_z"] = -_z(out["dxy"], 156)

    # === 7个DXY变化趋势变体 (全部预注册) ===
    # M1: 5日变化率
    out["M1_dxy_chg_5d"] = out["dxy"].pct_change(5)
    # M2: 20日变化率
    out["M2_dxy_chg_20d"] = out["dxy"].pct_change(20)
    # M3: 60日变化率
    out["M3_dxy_chg_60d"] = out["dxy"].pct_change(60)
    # M4: 20日绝对变化
    out["M4_dxy_diff_20"] = out["dxy"].diff(20)
    # M5: 均线比 (dxy/MA60)
    out["M5_dxy_ma_ratio"] = out["dxy"] / out["dxy"].rolling(60).mean()
    # M6: 突破信号 (dxy > 60日最高 × 0.95)
    high_60 = out["dxy"].rolling(60).max()
    out["M6_dxy_breakout"] = (out["dxy"] > high_60 * 0.95).astype(float)
    # M7: 趋势z-score (dxy - MA60 / std60)
    ma60 = out["dxy"].rolling(60).mean()
    std60 = out["dxy"].rolling(60).std()
    out["M7_dxy_trend_z"] = (out["dxy"] - ma60) / std60

    return out


def partial_ic(df: pd.DataFrame, factor: str, target: str = "fut_vol_20",
               controls: list = None) -> float:
    if controls is None:
        controls = ["vix_pct", "gvz_z"]
    sub = df[[factor, target] + controls].dropna()
    if len(sub) < 50:
        return np.nan
    X = np.column_stack([sub[c].values for c in controls] + [np.ones(len(sub))])
    y_factor = sub[factor].values
    beta_f, _, _, _ = np.linalg.lstsq(X, y_factor, rcond=None)
    rf = y_factor - X @ beta_f
    y_target = sub[target].values
    beta_t, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    rt = y_target - X @ beta_t
    return np.corrcoef(rf, rt)[0, 1]


def ic_raw(df: pd.DataFrame, factor: str, target: str = "fut_vol_5") -> float:
    sub = df[[factor, target]].dropna()
    if len(sub) < 50:
        return np.nan
    return sub[factor].corr(sub[target])


def subperiod_pic(df: pd.DataFrame, factor: str, target: str = "fut_vol_20") -> list:
    return [partial_ic(df[(df.index >= s) & (df.index <= e)], factor, target)
            for _, s, e in SUBPERIODS]


def main():
    print("=" * 60)
    print("DXY变化趋势与黄金关系研究 — Phase 1")
    print("=" * 60)

    df = build_factors(load_data())
    print(f"检验区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # 7个变体 + F10基线
    variants = [
        ("F10_baseline", "F10_dxy_z", "DXY水平z-score (基线)"),
        ("M1", "M1_dxy_chg_5d", "DXY 5日变化率"),
        ("M2", "M2_dxy_chg_20d", "DXY 20日变化率"),
        ("M3", "M3_dxy_chg_60d", "DXY 60日变化率"),
        ("M4", "M4_dxy_diff_20", "DXY 20日绝对变化"),
        ("M5", "M5_dxy_ma_ratio", "DXY/MA60 均线比"),
        ("M6", "M6_dxy_breakout", "DXY突破60日高点×0.95"),
        ("M7", "M7_dxy_trend_z", "DXY趋势z-score"),
    ]

    # === Phase 1: 单变体排名 ===
    print("\n=== Phase 1: 单变体排名 ===")
    header = "{:<5s}{:<18s}{:<28s}{:>10s}{:>10s}{:>8s}".format(
        "ID", "factor", "desc", "pIC_20d", "IC_5d", "subp>0")
    print(header)
    print("-" * len(header))

    results = []
    for vid, fc, desc in variants:
        pic_20 = partial_ic(df, fc)
        ic_5 = ic_raw(df, fc)
        sub_pics = subperiod_pic(df, fc)
        n_pos = sum(1 for x in sub_pics if not np.isnan(x) and x > 0)

        pic_str = f"{pic_20:+.4f}" if not np.isnan(pic_20) else "  --"
        ic_str = f"{ic_5:+.4f}" if not np.isnan(ic_5) else "  --"
        results.append({
            "id": vid, "factor": fc, "desc": desc, "pic_20": pic_20,
            "ic_5": ic_5, "sub_pics": sub_pics, "n_pos": n_pos,
        })
        print(f"{vid:<5s}{fc:<18s}{desc:<28s}{pic_str:>10s}{ic_str:>10s}{n_pos:>5d}/3")

    # === G4: 与F10相关系数 ===
    print("\n=== G4: 与F10_dxy_z相关性 (避免冗余) ===")
    f10_series = df["F10_dxy_z"].dropna()
    for r in results:
        if r["id"] == "F10_baseline":
            r["f10_corr"] = 1.0
            r["g4"] = False  # 自身不算
        else:
            corr = df[r["factor"]].corr(df["F10_dxy_z"])
            r["f10_corr"] = corr
            r["g4"] = abs(corr) < GATES["G4_f10_corr_max"]
        if r["id"] != "F10_baseline":
            g4_str = "✅" if r["g4"] else "❌"
            corr_str = f"{r['f10_corr']:+.3f}" if not np.isnan(r["f10_corr"]) else "  --"
            print(f"  {r['id']:<5s} corr={corr_str:>8s} G4(<0.7): {g4_str}")

    # === 全面判定 ===
    print("\n" + "=" * 60)
    print("综合判定 (G1-G4)")
    print("=" * 60)
    header = "{:<5s}{:<18s}{:>4s}{:>4s}{:>4s}{:>4s}{:>8s}".format(
        "ID", "factor", "G1", "G2", "G3", "G4", "通过数")
    print(header)
    print("-" * len(header))

    for r in results:
        if r["id"] == "F10_baseline":
            g1s = g2s = g3s = g4s = "—"
            n_pass_str = "基线"
        else:
            g1 = r["pic_20"] >= GATES["G1_pIC_20d_vol_min"] if not np.isnan(r["pic_20"]) else False
            g2 = r["n_pos"] >= 2
            g3 = r["ic_5"] >= GATES["G3_IC_5d_vol_min"] if not np.isnan(r["ic_5"]) else False
            g4 = r["g4"]
            r["g1"] = g1
            r["g2"] = g2
            r["g3"] = g3
            n_pass = sum([g1, g2, g3, g4])
            n_pass_str = f"{n_pass}/4"
            g1s = "✅" if g1 else "❌"
            g2s = "✅" if g2 else "❌"
            g3s = "✅" if g3 else "❌"
            g4s = "✅" if g4 else "❌"
        print(f"{r['id']:<5s}{r['factor']:<18s}{g1s:>4s}{g2s:>4s}{g3s:>4s}{g4s:>4s}{n_pass_str:>8s}")

    # === 与F10基线对比 (变体自身价值) ===
    print("\n=== 与F10基线对比 (变体能否独立贡献) ===")
    f10_pic = results[0]["pic_20"]
    print(f"F10_dxy_z (基线): pIC_20d = {f10_pic:+.4f}")
    for r in results[1:]:
        delta = r["pic_20"] - f10_pic
        stronger = "✅ 强于F10" if delta > 0.02 else ("≈ 相当" if abs(delta) <= 0.02 else "❌ 弱于F10")
        pic_str = f"{r['pic_20']:+.4f}" if not np.isnan(r["pic_20"]) else "  --"
        print(f"  {r['id']:<5s} pIC={pic_str} (Δ={delta:+.4f}) {stronger}")

    # === Phase 2: 组合检验 (若有≥2变体通过G1) ===
    g1_pass = [r for r in results[1:] if r["g1"]]
    print(f"\n通过G1的变体: {len(g1_pass)} 个 {[r['id'] for r in g1_pass]}")

    if len(g1_pass) >= 2:
        print("\n进入 Phase 2: 组合检验")

        # 找最强的变体
        best = max(g1_pass, key=lambda x: x["pic_20"] if not np.isnan(x["pic_20"]) else -999)

        # 5个预注册组合
        combos = [
            ("C1_F10", "F10_dxy_z", "F10基线"),
            (f"C2_{best['id']}", f"({best['factor']})", f"{best['id']}单独"),
            ("C3_F10+best", f"(F10_dxy_z + {best['factor']})", f"F10+{best['id']}等权"),
            ("C4_F10*best", f"(F10_dxy_z * {best['factor']})", f"F10*{best['id']}交互"),
            ("C5_0.7F10+0.3best", f"(0.7*F10_dxy_z + 0.3*{best['factor']})", f"权重F10主"),
        ]

        header = "{:<20s}{:<32s}{:>10s}{:>10s}{:>8s}".format(
            "ID", "formula", "pIC_20d", "IC_5d", "subp>0")
        print(header)
        print("-" * len(header))

        for cid, formula, desc in combos:
            df[f"_{cid}"] = df.eval(formula)
            pic_20 = partial_ic(df, f"_{cid}")
            ic_5 = ic_raw(df, f"_{cid}")
            sub_pics = subperiod_pic(df, f"_{cid}")
            n_pos = sum(1 for x in sub_pics if not np.isnan(x) and x > 0)
            pic_str = f"{pic_20:+.4f}" if not np.isnan(pic_20) else "  --"
            ic_str = f"{ic_5:+.4f}" if not np.isnan(ic_5) else "  --"
            print(f"{cid:<20s}{formula:<32s}{pic_str:>10s}{ic_str:>10s}{n_pos:>5d}/3")

        # 最佳组合判定
        combo_results = []
        for cid, formula, _ in combos:
            df[f"_{cid}"] = df.eval(formula)
            combo_results.append({
                "id": cid,
                "pic_20": partial_ic(df, f"_{cid}"),
                "sub_pics": subperiod_pic(df, f"_{cid}"),
                "n_pos": sum(1 for x in subperiod_pic(df, f"_{cid}")
                            if not np.isnan(x) and x > 0),
            })

        best_combo = max(combo_results[1:], key=lambda x: x["pic_20"]
                         if not np.isnan(x["pic_20"]) else -999)
        print(f"\n最佳组合: {best_combo['id']} pIC={best_combo['pic_20']:+.4f}")
        print(f"vs F10基线 +{best_combo['pic_20'] - f10_pic:.4f}")

        if best_combo["pic_20"] > f10_pic + 0.02:
            print(f"\n✅ 推荐: {best_combo['id']} 进入Phase 3集成原型")
        else:
            print(f"\n❌ 组合无实质提升, 不进入Phase 3")
    else:
        print("\n❌ 通过G1的变体不足, 不进入Phase 2, 归档负结果")


if __name__ == "__main__":
    main()