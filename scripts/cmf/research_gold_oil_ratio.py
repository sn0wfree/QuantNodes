#!/usr/bin/env python3
"""scripts/cmf/research_gold_oil_ratio.py — 金油比值风险因子研究.

纯只读脚本, 不修改DuckDB, 只输出研究结果.

预注册变体:
    R-zscore: 滚动z(比值), 窗口 126/252/504
    R-logdiff: 累计Δlog(gold)−Δlog(oil) 的z-score, 窗口252
    R-quantile: 比值的滚动百分位rank, 窗口252

检验:
    - IC/rankIC → 未来5/10/20日波动与收益
    - 五分位分层
    - 子周期 (2006-13/14-19/20-26)
    - 偏IC (控制VIX+GVZ) — Phase 3关口
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
GATES = {
    "G1_partial_IC_20d_vol_min": 0.10,
    "G2_subperiod_consistency_min": 2 / 3,
    "G3_window_robustness_max_sharpe_diff": 0.10,
}

SUBPERIODS = [
    ("2006-2013", "2006-01-01", "2013-12-31"),
    ("2014-2019", "2014-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
]


def load_data() -> pd.DataFrame:
    """加载金价(WTI + gold_spot USD) 和风控指标."""
    con = duckdb.connect(DB_PATH, read_only=True)
    gold = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_macro' AND value_col='gold_spot' ORDER BY trade_date"
    ).df()
    wti = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='fred_dcoil' AND value_col='wti' ORDER BY trade_date"
    ).df()
    vix = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    gvz = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='gvz' ORDER BY trade_date"
    ).df()
    con.close()

    for df in [gold, wti, vix, gvz]:
        df["trade_date"] = pd.to_datetime(df["trade_date"])

    df = pd.DataFrame({
        "gold_usd": gold.set_index("trade_date")["value"],
        "wti": wti.set_index("trade_date")["value"],
        "vix": vix.set_index("trade_date")["value"],
        "gvz": gvz.set_index("trade_date")["value"],
    }).sort_index().ffill().dropna()

    return df


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """预注册: 构建3种因子变体."""
    df["ratio"] = df["gold_usd"] / df["wti"]

    # R-zscore (3窗口)
    for w in [126, 252, 504]:
        mu = df["ratio"].rolling(w).mean()
        sd = df["ratio"].rolling(w).std()
        df[f"R_zscore_{w}"] = (df["ratio"] - mu) / sd

    # R-logdiff (累计log差, 252窗口z-score)
    df["logdiff"] = np.log(df["gold_usd"]) - np.log(df["wti"])
    df["cum_logdiff"] = df["logdiff"].rolling(252).sum()
    mu = df["cum_logdiff"].rolling(252).mean()
    sd = df["cum_logdiff"].rolling(252).std()
    df["R_logdiff_252"] = (df["cum_logdiff"] - mu) / sd

    # R-quantile (252窗口百分位rank)
    df["R_quantile_252"] = df["ratio"].rolling(252).rank(pct=True)

    # 现有风控指标
    df["vix_pct"] = df["vix"].rolling(252).rank(pct=True)
    df["gvz_z"] = (df["gvz"] - df["gvz"].rolling(90).mean()) / df["gvz"].rolling(90).std()

    # 未来K日波动与收益 (检验目标)
    df["gold_ret"] = df["gold_usd"].pct_change()
    for k in [5, 10, 20]:
        df[f"fut_ret_{k}"] = df["gold_usd"].shift(-k) / df["gold_usd"] - 1
        df[f"fut_vol_{k}"] = df["gold_ret"].shift(-1).rolling(k).std().shift(-(k - 1)) * np.sqrt(252)

    return df


def ic(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    """IC (Information Coefficient)."""
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < 50:
        return np.nan
    return sub["x"].corr(sub["y"], method=method)


def partial_ic(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """控制z后的偏IC."""
    sub = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(sub) < 50:
        return np.nan
    rx = sub["x"] - sub["z"] * np.corrcoef(sub["x"], sub["z"])[0, 1]
    ry = sub["y"] - sub["z"] * np.corrcoef(sub["y"], sub["z"])[0, 1]
    return np.corrcoef(rx, ry)[0, 1]


def quintile_analysis(df: pd.DataFrame, factor_col: str, target_col: str) -> pd.DataFrame:
    """五分位分层."""
    sub = df[[factor_col, target_col]].dropna()
    if len(sub) < 100:
        return pd.DataFrame()
    q = pd.qcut(sub[factor_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return sub.groupby(q, observed=True).agg(
        [(target_col, "mean"), (target_col, "count")]
    ).rename(columns={target_col: target_col})


def main():
    print("=" * 60)
    print("金油比值风险因子研究 — Phase 2+3 检验")
    print("=" * 60)

    df = build_factors(load_data())
    print(f"样本区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    factor_cols = ["R_zscore_126", "R_zscore_252", "R_zscore_504",
                   "R_logdiff_252", "R_quantile_252"]

    # ===== Phase 2: 基础IC检验 =====
    print("\n=== Phase 2: 各变体IC (全样本, 2006-2026) ===")
    print(f"{'因子':<18s}{'IC_5d_vol':>12s}{'IC_10d_vol':>12s}{'IC_20d_vol':>12s}"
          f"{'IC_5d_ret':>12s}{'IC_10d_ret':>12s}{'IC_20d_ret':>12s}")

    ic_table = {}
    for f in factor_cols:
        row = []
        for k in [5, 10, 20]:
            row.append(ic(df[f], df[f"fut_vol_{k}"]))
            row.append(ic(df[f], df[f"fut_ret_{k}"]))
        ic_table[f] = row
        formatted = [f"{v:+.3f}" if not np.isnan(v) else "  --" for v in row]
        # 顺序: vol5, ret5, vol10, ret10, vol20, ret20 (因为row按k=5,10,20分别加vol+ret)
        # 重新整理为 (vol5, vol10, vol20, ret5, ret10, ret20)
        reordered = [row[0], row[2], row[4], row[1], row[3], row[5]]
        formatted = [f"{v:+.3f}" if not np.isnan(v) else "  --" for v in reordered]
        print(f"{f:<18s}{formatted[0]:>12s}{formatted[1]:>12s}{formatted[2]:>12s}"
              f"{formatted[3]:>12s}{formatted[4]:>12s}{formatted[5]:>12s}")

    # ===== Phase 3: 偏IC (控制VIX+GVZ) =====
    print("\n=== Phase 3: 偏IC (控制VIX_pct + GVZ_z) → 未来波动 ===")
    print(f"{'因子':<18s}{'pIC_20d_vol':>15s}{'G1(>=0.10)':>15s}")
    partial_table = {}
    for f in factor_cols:
        pic = partial_ic(df[f], df["fut_vol_20"], df["vix_pct"])
        # 同时控制两个变量 (简化: 与第一个的残差)
        from numpy.linalg import lstsq
        sub = df[[f, "fut_vol_20", "vix_pct", "gvz_z"]].dropna()
        if len(sub) > 50:
            X = np.column_stack([sub["vix_pct"].values, sub["gvz_z"].values, np.ones(len(sub))])
            y = sub[f].values
            beta, _, _, _ = lstsq(X, y, rcond=None)
            residual = y - X @ beta
            y_target = sub["fut_vol_20"].values
            X_target = np.column_stack([sub["vix_pct"].values, sub["gvz_z"].values, np.ones(len(sub))])
            beta_t, _, _, _ = lstsq(X_target, y_target, rcond=None)
            residual_t = y_target - X_target @ beta_t
            pic2 = np.corrcoef(residual, residual_t)[0, 1]
        else:
            pic2 = np.nan
        partial_table[f] = pic2
        g1 = "✅" if pic2 >= 0.10 else "❌"
        print(f"{f:<18s}{pic2:>+15.4f}{g1:>15s}")

    # ===== G2: 子周期稳健性 =====
    print("\n=== G2: 子周期方向一致性 (偏IC > 0 计数) ===")
    subperiod_results = {f: [] for f in factor_cols}
    for label, s, e in SUBPERIODS:
        mask = (df.index >= s) & (df.index <= e)
        sub = df[mask]
        print(f"\n  子周期 {label}:")
        print(f"  {'因子':<18s}{'pIC_20d_vol':>15s}")
        for f in factor_cols:
            from numpy.linalg import lstsq
            ss = sub[[f, "fut_vol_20", "vix_pct", "gvz_z"]].dropna()
            if len(ss) < 50:
                pic = np.nan
            else:
                X = np.column_stack([ss["vix_pct"].values, ss["gvz_z"].values, np.ones(len(ss))])
                y = ss[f].values
                beta, _, _, _ = lstsq(X, y, rcond=None)
                r1 = y - X @ beta
                X2 = np.column_stack([ss["vix_pct"].values, ss["gvz_z"].values, np.ones(len(ss))])
                y2 = ss["fut_vol_20"].values
                beta2, _, _, _ = lstsq(X2, y2, rcond=None)
                r2 = y2 - X2 @ beta2
                pic = np.corrcoef(r1, r2)[0, 1]
            subperiod_results[f].append(pic)
            print(f"  {f:<18s}{pic:>+15.4f}")

    print("\n  汇总 (>0 子周期数 / 3):")
    for f in factor_cols:
        n_pos = sum(1 for x in subperiod_results[f] if not np.isnan(x) and x > 0)
        g2 = "✅" if n_pos >= 2 else "❌"
        print(f"    {f:<18s}  {n_pos}/3 {g2}")

    # ===== G3: 窗口稳健性 (zscore三档对比) =====
    print("\n=== G3: z窗口稳健性 (126 vs 252 vs 504) ===")
    print(f"  {'窗口':<10s}{'pIC_20d_vol':>15s}")
    ws_pic = []
    for w in [126, 252, 504]:
        from numpy.linalg import lstsq
        f = f"R_zscore_{w}"
        ss = df[[f, "fut_vol_20", "vix_pct", "gvz_z"]].dropna()
        X = np.column_stack([ss["vix_pct"].values, ss["gvz_z"].values, np.ones(len(ss))])
        y = ss[f].values
        beta, _, _, _ = lstsq(X, y, rcond=None)
        r1 = y - X @ beta
        X2 = X
        y2 = ss["fut_vol_20"].values
        beta2, _, _, _ = lstsq(X2, y2, rcond=None)
        r2 = y2 - X2 @ beta2
        pic = np.corrcoef(r1, r2)[0, 1]
        ws_pic.append(pic)
        print(f"  {w:<10d}{pic:>+15.4f}")

    ws_max_diff = max(ws_pic) - min(ws_pic)
    g3 = "✅" if ws_max_diff < 0.10 else "❌"
    print(f"  极差: {ws_max_diff:.4f} {g3} (门槛 <0.10)")

    # ===== 关口判定 =====
    print("\n" + "=" * 60)
    print("关口判定")
    print("=" * 60)

    # G1: 用主因子 (R_zscore_252) 的偏IC
    g1_pic = partial_table["R_zscore_252"]
    g1_pass = g1_pic >= GATES["G1_partial_IC_20d_vol_min"]
    print(f"G1 偏IC(20日波动) >= 0.10: R_zscore_252 pIC={g1_pic:+.4f} {'✅' if g1_pass else '❌'}")

    # G2: 主因子子周期
    g2_n_pos = sum(1 for x in subperiod_results["R_zscore_252"] if not np.isnan(x) and x > 0)
    g2_pass = g2_n_pos >= 2
    print(f"G2 子周期>=2of3: R_zscore_252 = {g2_n_pos}/3 {'✅' if g2_pass else '❌'}")

    # G3: 窗口极差
    g3_pass = ws_max_diff < GATES["G3_window_robustness_max_sharpe_diff"]
    print(f"G3 窗口极差 < 0.10: {ws_max_diff:.4f} {'✅' if g3_pass else '❌'}")

    all_pass = g1_pass and g2_pass and g3_pass
    print(f"\n最终判定: {'✅ ALL PASS — 可议集成' if all_pass else '❌ 部分失败 — 归档负结果'}")

    # ===== 五分位分层 (Phase 2 辅助展示) =====
    print("\n=== Phase 2辅助: R_zscore_252 五分位分层 ===")
    print(f"{'分位':<6s}{'fut_ret_20d':>15s}{'fut_vol_20d':>15s}{'样本数':>10s}")
    sub = df[["R_zscore_252", "fut_ret_20", "fut_vol_20"]].dropna()
    q = pd.qcut(sub["R_zscore_252"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    grp = sub.groupby(q, observed=True).agg(
        fut_ret_20d=("fut_ret_20", "mean"),
        fut_vol_20d=("fut_vol_20", "mean"),
        n=("R_zscore_252", "count"),
    )
    for idx, row in grp.iterrows():
        print(f"{idx:<6s}{row['fut_ret_20d']:>+15.2%}{row['fut_vol_20d']:>15.2%}{int(row['n']):>10d}")


if __name__ == "__main__":
    main()