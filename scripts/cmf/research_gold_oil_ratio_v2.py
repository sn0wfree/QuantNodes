#!/usr/bin/env python3
"""scripts/cmf/research_gold_oil_ratio_v2.py — Regime-adaptive二期研究.

纯只读脚本, 不修改DuckDB.
测试4个regime指标 + 交互项能否让R_zscore跨regime方向一致.

预注册:
    R1: RIR趋势 (TIPS.diff(252) > 0)
    R2: VIX百分位 > 0.7
    R3: CPI同比 > 3%
    R4: Fed Rate方向 (fed_rate.diff(252) > 0)

门槛:
    - 三个子周期偏IC全正 (vs 原2/3)
    - 全样本偏IC ≥ 0.10
    - 至少2/4个regime达标
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
    ("2008-2013", "2008-06-03", "2013-12-31"),
    ("2014-2019", "2014-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
]


def load_data() -> pd.DataFrame:
    """加载金价/WTI/4个regime指标."""
    con = duckdb.connect(DB_PATH, read_only=True)
    queries = {
        "gold": ("gold_macro", "gold_spot"),
        "wti": ("fred_dcoil", "wti"),
        "tips": ("gold_fred", "tips"),
        "vix": ("gold_fred", "vix"),
        "fed": ("gold_macro", "fed_rate"),
        "rir": ("gold_fred", "rir"),
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造因子和regime指标."""
    df["ratio"] = df["gold"] / df["wti"]
    df["R_zscore_252"] = (
        (df["ratio"] - df["ratio"].rolling(252).mean())
        / df["ratio"].rolling(252).std()
    )
    df["gold_ret"] = df["gold"].pct_change()
    df["fut_vol_20"] = (
        df["gold_ret"].shift(-1).rolling(20).std().shift(-19) * np.sqrt(252)
    )
    df["vix_pct"] = df["vix"].rolling(252).rank(pct=True)

    # 4个regime指标
    # R1: RIR趋势 (TIPS.diff(252) > 0)
    df["R1_indicator"] = (df["tips"].ffill().diff(252) > 0).astype(float)
    # R2: VIX百分位 > 0.7
    df["R2_indicator"] = (df["vix_pct"] > 0.7).astype(float)
    # R3: CPI同比 > 3% (用tips年同比近似, 仅做演示)
    df["R3_indicator"] = (df["tips"].pct_change(252) > 0.03).astype(float)
    # R4: Fed Rate方向 (fed_rate.diff(252) > 0)
    df["R4_indicator"] = (df["fed"].diff(252) > 0).astype(float)

    return df


def partial_ic_factor(df: pd.DataFrame, factor: str, target: str = "fut_vol_20",
                     controls: list = None) -> float:
    """单因子或regime交互项的偏IC."""
    if controls is None:
        controls = ["vix_pct"]
    sub = df[[factor, target] + controls].dropna()
    if len(sub) < 50:
        return np.nan
    # 控制 controls
    X = np.column_stack([sub[c].values for c in controls] + [np.ones(len(sub))])
    y_factor = sub[factor].values
    beta_f, _, _, _ = np.linalg.lstsq(X, y_factor, rcond=None)
    rf = y_factor - X @ beta_f
    y_target = sub[target].values
    beta_t, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    rt = y_target - X @ beta_t
    return np.corrcoef(rf, rt)[0, 1]


def test_regime(df: pd.DataFrame, factor: str, regime_col: str, label: str):
    """测试单个regime-adaptive因子."""
    # 构造自适应因子: regime=1 → 原符号, regime=0 → 反转
    adaptive = factor + f" × adapt({regime_col})"

    # 对每个子周期计算偏IC (控制vix_pct)
    pic_full = partial_ic_factor(df, factor)
    # 子周期
    sub_pics = []
    for _, s, e in SUBPERIODS:
        mask = (df.index >= s) & (df.index <= e)
        sub = df[mask]
        pic = partial_ic_factor(sub, factor)
        sub_pics.append(pic)

    # 自适应因子: regime=0时反转符号
    df["adaptive_factor"] = df[factor].where(df[regime_col] == 1, -df[factor])
    adaptive_pic_full = partial_ic_factor(df, "adaptive_factor")
    adaptive_sub_pics = []
    for _, s, e in SUBPERIODS:
        mask = (df.index >= s) & (df.index <= e)
        sub = df[mask].copy()
        pic = partial_ic_factor(sub, "adaptive_factor")
        adaptive_sub_pics.append(pic)

    # 输出
    n_pos_orig = sum(1 for x in sub_pics if not np.isnan(x) and x > 0)
    n_pos_adapt = sum(1 for x in adaptive_sub_pics if not np.isnan(x) and x > 0)

    print(f"\n--- {label} ---")
    print(f"原因子 ({factor}):")
    print(f"  全样本偏IC: {pic_full:+.4f}")
    print(f"  子周期偏IC: {' '.join(f'{p:+.3f}' for p in sub_pics)}")
    print(f"  正向子周期数: {n_pos_orig}/3")
    print(f"自适应 ({adaptive}):")
    print(f"  全样本偏IC: {adaptive_pic_full:+.4f}")
    print(f"  子周期偏IC: {' '.join(f'{p:+.3f}' for p in adaptive_sub_pics)}")
    print(f"  正向子周期数: {n_pos_adapt}/3")

    # 门槛检查
    g1_pass = adaptive_pic_full >= 0.10
    g2_pass = n_pos_adapt == 3  # 全部正向
    g3_pass = adaptive_pic_full >= 0.08  # 略宽松,因自适应应比原更强

    print(f"  G1(偏IC>=0.10): {'✅' if g1_pass else '❌'}")
    print(f"  G2(3/3正向): {'✅' if g2_pass else '❌'}")
    overall = g1_pass and g2_pass
    print(f"  综合: {'✅ 双门槛通过' if overall else '❌ 未通过'}")

    return {
        "label": label,
        "orig_pic_full": pic_full,
        "orig_sub_pics": sub_pics,
        "orig_n_pos": n_pos_orig,
        "adapt_pic_full": adaptive_pic_full,
        "adapt_sub_pics": adaptive_sub_pics,
        "adapt_n_pos": n_pos_adapt,
        "g1": g1_pass,
        "g2": g2_pass,
    }


def main():
    print("=" * 60)
    print("金油比值二期研究 — Regime-Adaptive")
    print("=" * 60)

    df = build_features(load_data())
    print(f"样本区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # 4个regime指标并行测试
    results = []
    results.append(test_regime(df, "R_zscore_252", "R1_indicator", "R1: RIR趋势 (TIPS.diff(252)>0)"))
    results.append(test_regime(df, "R_zscore_252", "R2_indicator", "R2: VIX百分位>0.7"))
    results.append(test_regime(df, "R_zscore_252", "R3_indicator", "R3: CPI同比>3% (tips同比近似)"))
    results.append(test_regime(df, "R_zscore_252", "R4_indicator", "R4: Fed Rate方向"))

    # 汇总判定
    print("\n" + "=" * 60)
    print("判定汇总 (门槛: 至少2/4个regime双门槛通过)")
    print("=" * 60)

    n_pass = 0
    for r in results:
        if r["g1"] and r["g2"]:
            n_pass += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {r['label']:<40s} {status}")

    if n_pass >= 2:
        print(f"\n最终判定: ✅ 至少2个regime通过, regime-adaptive有研究价值 (n_pass={n_pass})")
    else:
        print(f"\n最终判定: ❌ 通过regime不足, 与Phase 1并列归档负结果 (n_pass={n_pass})")


if __name__ == "__main__":
    main()