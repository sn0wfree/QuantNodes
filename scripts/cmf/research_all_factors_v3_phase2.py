#!/usr/bin/env python3
"""scripts/cmf/research_all_factors_v3_phase2.py — Phase 2 组合检验.

基于Phase 1筛选的最强因子 (F10_dxy_z 4/4过, F5_epu_pct 3/4过) 做两两/三三组合,
检验组合是否能进一步提升偏IC.

预注册组合:
    P1: F10_dxy_z (基线, 4/4过)
    P2: F5_epu_pct (3/4过)
    P3: F2_us10y_z (2/4过)
    P4: F10 + F5 (最优候选对)
    P5: F10 + F2
    P6: F10 + F5 + F2
    P7: (F10 + F5) / 2 等权
    P8: 0.5*F10 + 0.3*F5 + 0.2*F2 (权重方案)
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
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2025", "2023-01-01", "2025-12-31"),
]


def load_data() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    queries = {
        "gold": ("gold_macro", "gold_spot"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "epu": ("gold_fred", "epu"),
        "us10y": ("gold_macro", "us_10y"),
        "dxy": ("gold_macro", "dxy"),
    }
    dfs = {}
    for col, (src, vc) in queries.items():
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
    out = df.copy()
    out["gold_ret"] = out["gold"].pct_change()
    for k in [5, 20]:
        out[f"fut_vol_{k}"] = out["gold_ret"].shift(-1).rolling(k).std().shift(-(k - 1)) * np.sqrt(252)
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)
    out["F2_us10y_z"] = -_z(out["us10y"].diff(13), 156)
    out["F5_epu_pct"] = out["epu"].rolling(252).rank(pct=True)
    out["F10_dxy_z"] = -_z(out["dxy"], 156)
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
    print("全因子组合检验 — Phase 2")
    print("=" * 60)

    df = build_factors(load_data())["2018":"2025"]
    print(f"检验区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # 8个预注册组合
    combos = [
        ("P1_F10", "F10_dxy_z", "基线"),
        ("P2_F5", "F5_epu_pct", "3/4过"),
        ("P3_F2", "F2_us10y_z", "2/4过"),
        ("P4_F10+F5", "(F10_dxy_z+F5_epu_pct)", "等权最优候选对"),
        ("P5_F10+F2", "(F10_dxy_z+F2_us10y_z)", "等权"),
        ("P6_F10+F5+F2", "(F10_dxy_z+F5_epu_pct+F2_us10y_z)", "三因子等权"),
        ("P7_weighted", "0.5*F10_dxy_z+0.3*F5_epu_pct+0.2*F2_us10y_z", "权重方案"),
        ("P8_multiply", "F10_dxy_z*F5_epu_pct", "交互项 (乘法)"),
    ]

    header = "{:<15s}{:<32s}{:>10s}{:>10s}{:>10s}{:>8s}".format(
        "ID", "formula", "pIC_20d", "IC_5d", "vix_corr", "subp>0")
    print("\n=== 组合因子排名 ===")
    print(header)
    print("-" * len(header))

    results = []
    for cid, formula, desc in combos:
        df[f"_{cid}"] = df.eval(formula)
        fac_col = f"_{cid}"
        pic_20 = partial_ic(df, fac_col)
        ic_5 = ic_raw(df, fac_col)
        vix_corr = df[fac_col].corr(df["vix_pct"])
        sub_pics = subperiod_pic(df, fac_col)
        n_pos = sum(1 for x in sub_pics if not np.isnan(x) and x > 0)

        pic_str = f"{pic_20:+.4f}" if not np.isnan(pic_20) else "  --"
        ic_str = f"{ic_5:+.4f}" if not np.isnan(ic_5) else "  --"
        vix_str = f"{vix_corr:+.3f}" if not np.isnan(vix_corr) else "  --"

        results.append({
            "id": cid, "desc": desc, "pic_20": pic_20, "ic_5": ic_5,
            "vix_corr": vix_corr, "sub_pics": sub_pics, "n_pos": n_pos,
        })
        print(f"{cid:<15s}{formula:<32s}{pic_str:>10s}{ic_str:>10s}{vix_str:>10s}{n_pos:>5d}/3")

    # 详细子周期
    print("\n=== 子周期详情 ===")
    print(f"{'ID':<15s}{'2018-20':>10s}{'2021-22':>10s}{'2023-25':>10s}{'>0':>6s}")
    for r in results:
        sub_strs = [f"{x:+.3f}" if not np.isnan(x) else "  --" for x in r["sub_pics"]]
        print(f"{r['id']:<15s}{sub_strs[0]:>10s}{sub_strs[1]:>10s}{sub_strs[2]:>10s}"
              f"{r['n_pos']:>5d}/3")

    # 与Phase1的F10比较 (基线提升)
    print("\n=== 与Phase1 F10基线比较 ===")
    f10_pic = results[0]["pic_20"]
    print(f"F10_dxy_z (基线): pIC_20d = {f10_pic:+.4f}")
    for r in results[1:]:
        if r["pic_20"] > f10_pic + 0.01:
            verdict = "✅ 提升>1pp"
        elif r["pic_20"] > f10_pic:
            verdict = "⚠️ 微提升<1pp"
        else:
            verdict = "❌ 无提升"
        delta = r["pic_20"] - f10_pic
        print(f"{r['id']:<15s} pIC={r['pic_20']:+.4f} (Δ={delta:+.4f}) {verdict}")

    # Go/No-Go
    print("\n" + "=" * 60)
    print("Phase 2 判定 (Go/No-Go)")
    print("=" * 60)

    # 门槛: 组合pIC需超过基线F10至少1pp, 且子周期3/3通过
    best = max(results[1:], key=lambda x: x["pic_20"] if not np.isnan(x["pic_20"]) else -999)
    print(f"最佳组合: {best['id']} pIC={best['pic_20']:+.4f} (vs F10基线 {f10_pic:+.4f})")
    print(f"提升: {best['pic_20'] - f10_pic:+.4f}")

    if best["pic_20"] > f10_pic + 0.01 and best["n_pos"] == 3:
        print(f"\n✅ 推荐: {best['id']} 进入Phase 3集成原型")
    elif best["pic_20"] > f10_pic:
        print(f"\n⚠️ 微提升, 但提升<1pp, 不议集成")
    else:
        print(f"\n❌ 组合无提升, F10单独即可, 不进入Phase 3")


if __name__ == "__main__":
    main()