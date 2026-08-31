#!/usr/bin/env python3
"""scripts/cmf/research_all_factors_v3.py — 全因子单因子研究 Phase 1.

纯只读脚本, 不修改DuckDB.
检验11个因子的独立预测力 (偏IC控制VIX+GVZ) + 子周期稳健性 + 时效性 + 与VIX相关性.

预注册门槛:
    G1: 偏IC(20日波动) ≥ +0.08
    G2: ≥2/3子周期偏IC > 0
    G3: IC(5日波动) ≥ 0.05
    G4: 与VIX相关系数 < 0.5
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

GATES = {
    "G1_pIC_20d_vol_min": 0.08,
    "G2_subperiod_min": 2 / 3,
    "G3_IC_5d_vol_min": 0.05,
    "G4_vix_corr_max": 0.5,
}


def load_data() -> pd.DataFrame:
    """加载11个因子的原始数据."""
    con = duckdb.connect(DB_PATH, read_only=True)
    queries = {
        "gold": ("gold_macro", "gold_spot"),
        "tips": ("gold_fred", "tips"),
        "us10y": ("gold_macro", "us_10y"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "gprd": ("gold_risk", "gprd"),
        "epu": ("gold_fred", "epu"),
        "comex_inv": ("gold_position", "comex_inv"),
        "cftc_net": ("gold_position", "cftc_net"),
        "etf_tonnes": ("gold_position", "etf_tonnes"),
        "cb_reserves": ("gold_position", "cb_reserves"),
        "dxy": ("gold_macro", "dxy"),
        "fed": ("gold_macro", "fed_rate"),
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

    df = pd.DataFrame(dfs).sort_index().ffill().dropna()
    return df


def _z(s: pd.Series, window: int = 156) -> pd.Series:
    """滚动z-score标准化 (156周 = 3年)."""
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """构建11个预注册因子."""
    out = df.copy()
    out["gold_ret"] = out["gold"].pct_change()

    # 未来K日波动与收益 (shift2与生产口径一致)
    for k in [5, 20]:
        out[f"fut_vol_{k}"] = (
            out["gold_ret"].shift(-1).rolling(k).std().shift(-(k - 1)) * np.sqrt(252)
        )
        out[f"fut_ret_{k}"] = out["gold"].shift(-k) / out["gold"] - 1

    # VIX百分位 + GVZ zscore (作为控制变量)
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)

    # === L1 利率层 (权重25%) ===
    # F1: -z(tips.diff(13), 156w) - 实际利率上行→黄金弱 (取负)
    out["F1_tips_z"] = -_z(out["tips"].diff(13), 156)
    # F2: -z(us10y.diff(13), 156w) - 美债收益率上行→黄金弱
    out["F2_us10y_z"] = -_z(out["us10y"].diff(13), 156)

    # === L2 风险情绪层 (权重20%) ===
    # F3: vix百分位 (已存在)
    out["F3_vix_pct"] = out["vix_pct"]
    # F4: gprd百分位
    out["F4_gprd_pct"] = out["gprd"].rolling(252).rank(pct=True)
    # F5: epu百分位
    out["F5_epu_pct"] = out["epu"].rolling(252).rank(pct=True)

    # === L3 实体需求层 (权重20%, 重点) ===
    # F6: -z(comex_inv, 156w) - 库存下降=需求强
    out["F6_comex_inv_z"] = -_z(out["comex_inv"], 156)
    # F7: z(cftc.pct_change(13), 156w) - 投机多头增仓
    out["F7_cftc_chg_z"] = _z(out["cftc_net"].pct_change(13), 156)
    # F8: z(etf.pct_change(4), 156w) - ETF流入
    out["F8_etf_chg_z"] = _z(out["etf_tonnes"].pct_change(4), 156)
    # F9: z(cb.pct_change(26), 156w) - 央行购金
    out["F9_cb_chg_z"] = _z(out["cb_reserves"].pct_change(26), 156)

    # === L4 美元层 (权重10%) ===
    # F10: -z(dxy, 156w) - 美元弱=黄金强
    out["F10_dxy_z"] = -_z(out["dxy"], 156)

    # === L5 regime层 (权重25%) ===
    # F11: z(fed_rate, 156w)
    out["F11_fed_z"] = _z(out["fed"], 156)

    return out


def partial_ic(df: pd.DataFrame, factor: str, target: str = "fut_vol_20",
               controls: list = None) -> float:
    """控制变量后的偏IC."""
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


def ic(df: pd.DataFrame, factor: str, target: str = "fut_vol_5") -> float:
    """原始IC."""
    sub = df[[factor, target]].dropna()
    if len(sub) < 50:
        return np.nan
    return sub[factor].corr(sub[target])


def subperiod_pic(df: pd.DataFrame, factor: str, target: str = "fut_vol_20") -> list:
    """三个子周期的偏IC."""
    results = []
    for _, s, e in SUBPERIODS:
        mask = (df.index >= s) & (df.index <= e)
        sub = df[mask]
        results.append(partial_ic(sub, factor, target))
    return results


def main():
    print("=" * 60)
    print("全因子单因子研究 — Phase 1")
    print("=" * 60)

    df = build_factors(load_data())
    # 限制检验区间
    df = df["2018":"2025"].copy()
    print(f"检验区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # 11个因子
    factor_cols = [
        ("F1", "F1_tips_z", "L1利率"),
        ("F2", "F2_us10y_z", "L1利率"),
        ("F3", "F3_vix_pct", "L2风险"),
        ("F4", "F4_gprd_pct", "L2风险"),
        ("F5", "F5_epu_pct", "L2风险"),
        ("F6", "F6_comex_inv_z", "L3实体"),
        ("F7", "F7_cftc_chg_z", "L3实体"),
        ("F8", "F8_etf_chg_z", "L3实体"),
        ("F9", "F9_cb_chg_z", "L3实体"),
        ("F10", "F10_dxy_z", "L4美元"),
        ("F11", "F11_fed_z", "L5regime"),
    ]

    # === 单因子排名表 ===
    print("\n=== 单因子排名 ===")
    header = "{:<5s}{:<18s}{:<10s}{:>10s}{:>10s}{:>10s}{:>6s}{:>6s}{:>8s}".format(
        "ID", "factor", "layer", "pIC_20d", "IC_5d", "vix_corr", "G1", "G3", "subp>0")
    print(header)
    print("-" * len(header))

    results = []
    for fid, fc, layer in factor_cols:
        pic_20 = partial_ic(df, fc, "fut_vol_20")
        ic_5 = ic(df, fc, "fut_vol_5")
        vix_corr = df[fc].corr(df["vix_pct"])
        sub_pics = subperiod_pic(df, fc)
        n_pos = sum(1 for x in sub_pics if not np.isnan(x) and x > 0)

        g1 = "✅" if not np.isnan(pic_20) and pic_20 >= GATES["G1_pIC_20d_vol_min"] else "❌"
        g3 = "✅" if not np.isnan(ic_5) and ic_5 >= GATES["G3_IC_5d_vol_min"] else "❌"

        results.append({
            "id": fid, "factor": fc, "layer": layer,
            "pic_20": pic_20, "ic_5": ic_5, "vix_corr": vix_corr,
            "sub_pics": sub_pics, "n_pos": n_pos,
            "g1": g1 == "✅", "g3": g3 == "✅",
        })

        vix_corr_str = f"{vix_corr:+.3f}" if not np.isnan(vix_corr) else "  --"
        pic_20_str = f"{pic_20:+.4f}" if not np.isnan(pic_20) else "  --"
        ic_5_str = f"{ic_5:+.4f}" if not np.isnan(ic_5) else "  --"
        print("{:<5s}{:<18s}{:<10s}{:>10s}{:>10s}{:>10s}{:>6s}{:>6s}{:>5d}/3".format(
            fid, fc, layer, pic_20_str, ic_5_str, vix_corr_str, g1, g3, n_pos))

    # === G4: 与VIX低相关 (排除冗余) ===
    print("\n=== G4: 与VIX低相关 (因子冗余度检验) ===")
    print(f"{'ID':<5s}{'factor':<18s}{'vix_corr':>10s}{'G4(<0.5)':>12s}")
    for r in results:
        g4 = "✅" if abs(r["vix_corr"]) < GATES["G4_vix_corr_max"] else "❌"
        r["g4"] = g4 == "✅"
        vix_corr_str = f"{r['vix_corr']:+.3f}" if not np.isnan(r["vix_corr"]) else "  --"
        print(f"{r['id']:<5s}{r['factor']:<18s}{vix_corr_str:>10s}{g4:>12s}")

    # === G2: 子周期稳健性 (各因子的子周期偏IC详情) ===
    print("\n=== G2: 子周期偏IC详情 ===")
    print(f"{'ID':<5s}{'factor':<18s}{'2018-20':>10s}{'2021-22':>10s}{'2023-25':>10s}{'>0子周期':>10s}{'G2':>5s}")
    for r in results:
        g2 = "✅" if r["n_pos"] >= 2 else "❌"
        r["g2"] = g2 == "✅"
        sub_strs = [f"{x:+.3f}" if not np.isnan(x) else "  --" for x in r["sub_pics"]]
        print(f"{r['id']:<5s}{r['factor']:<18s}{sub_strs[0]:>10s}{sub_strs[1]:>10s}{sub_strs[2]:>10s}"
              f"{r['n_pos']:>5d}/3{g2:>5s}")

    # === 综合判定 ===
    print("\n" + "=" * 60)
    print("综合判定 (Go/No-Go)")
    print("=" * 60)
    header = "{:<5s}{:<18s}{:>4s}{:>4s}{:>4s}{:>4s}{:>8s}".format(
        "ID", "factor", "G1", "G2", "G3", "G4", "通过数")
    print(header)
    print("-" * len(header))
    for r in results:
        n_pass = sum([r["g1"], r["g2"], r["g3"], r["g4"]])
        g1s = "✅" if r["g1"] else "❌"
        g2s = "✅" if r["g2"] else "❌"
        g3s = "✅" if r["g3"] else "❌"
        g4s = "✅" if r["g4"] else "❌"
        print(f"{r['id']:<5s}{r['factor']:<18s}{g1s:>4s}{g2s:>4s}{g3s:>4s}{g4s:>4s}{n_pass:>5d}/4")

    # 全部4过
    full_pass = [r for r in results if sum([r["g1"], r["g2"], r["g3"], r["g4"]]) == 4]
    # 3过 (高级潜在)
    high_pass = [r for r in results if sum([r["g1"], r["g2"], r["g3"], r["g4"]]) == 3]

    print(f"\n4/4门槛通过: {len(full_pass)} 个 {[r['id'] for r in full_pass]}")
    print(f"3/4门槛通过: {len(high_pass)} 个 {[r['id'] for r in high_pass]}")

    if full_pass:
        print(f"\n✅ 推荐进入Phase 2-3: 最强候选因子={[r['id'] for r in full_pass]}")
    elif high_pass:
        print(f"\n⚠️ 候选待续: {len(high_pass)} 个因子3/4过, 需组合或降低门槛")
    else:
        print(f"\n❌ 全因子单因子无独立预测力, 归档负结果")

    # === 因子间相关性矩阵 (L3层重点) ===
    print("\n=== 因子间相关性 (关键层) ===")
    key_factors = ["F3_vix_pct", "F4_gprd_pct", "F5_epu_pct",
                   "F6_comex_inv_z", "F7_cftc_chg_z", "F8_etf_chg_z", "F9_cb_chg_z",
                   "F10_dxy_z", "F11_fed_z"]
    corr = df[key_factors].corr().round(2)
    print(corr.to_string())


if __name__ == "__main__":
    main()