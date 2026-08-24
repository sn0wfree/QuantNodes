# coding=utf-8
"""scripts/cmf/14_carry_param_optimization.py

Carry参数优化网格搜索

测试范围:
    z_window: [26, 39, 52, 65, 78]
    mom_window: [4, 8, 13, 20, 26]
    min_duration: [1, 2, 4, 6, 8]
"""
import sys
sys.path.insert(0, ".")

import duckdb
import numpy as np
import pandas as pd
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import (
    build_signal_carry, backtest_carry,
)

DB = "data/cmf/fundamental.duckdb"


def run_grid_search():
    """执行参数网格搜索."""
    con = duckdb.connect(DB, read_only=True)

    results = []

    # 参数范围
    z_windows = [26, 39, 52, 65, 78]
    mom_windows = [4, 8, 13, 20, 26]
    min_durations = [1, 2, 4, 6, 8]

    total = len(z_windows) * len(mom_windows) * len(min_durations)
    print(f"开始网格搜索: {total}种组合")
    print("=" * 70)

    for i, zw in enumerate(z_windows):
        for mw in mom_windows:
            for md in min_durations:
                try:
                    sig = build_signal_carry(
                        con, variant="state_mom_curv2",
                        z_window=zw, mom_window=mw, min_duration=md,
                    )
                    r = backtest_carry(sig)

                    results.append({
                        "z_window": zw,
                        "mom_window": mw,
                        "min_duration": md,
                        "ann_ret": r["ann_ret"] * 100,
                        "sharpe": r["sharpe"],
                        "max_dd": r["max_dd"] * 100,
                        "win_rate": r["win_rate"] * 100,
                        "pct_active": r["pct_active"],
                    })
                except Exception as e:
                    print(f"  错误: zw={zw}, mw={mw}, md={md}: {e}")

    df = pd.DataFrame(results)
    return df


def analyze_results(df):
    """分析结果."""
    print("\n" + "=" * 70)
    print("参数优化结果分析")
    print("=" * 70)

    # Top 10 by Sharpe
    print("\nTop 10 Sharpe:")
    print("-" * 70)
    top_sharpe = df.nlargest(10, "sharpe")
    print(top_sharpe[["z_window", "mom_window", "min_duration", "ann_ret", "sharpe", "max_dd", "win_rate"]].to_string(index=False))

    # Top 10 by MaxDD (least negative)
    print("\nTop 10 MaxDD (最小回撤):")
    print("-" * 70)
    top_dd = df.nlargest(10, "max_dd")  # max_dd is negative, so larger = less negative
    print(top_dd[["z_window", "mom_window", "min_duration", "ann_ret", "sharpe", "max_dd", "win_rate"]].to_string(index=False))

    # Top 10 by combined score (Sharpe + abs(MaxDD)/100)
    df["score"] = df["sharpe"] + df["max_dd"].abs() / 100  # penalize large drawdowns
    print("\nTop 10 综合评分 (Sharpe + MaxDD惩罚):")
    print("-" * 70)
    top_combined = df.nlargest(10, "score")
    print(top_combined[["z_window", "mom_window", "min_duration", "ann_ret", "sharpe", "max_dd", "win_rate", "score"]].to_string(index=False))

    # 单参数敏感性
    print("\n" + "=" * 70)
    print("单参数敏感性")
    print("=" * 70)

    for param in ["z_window", "mom_window", "min_duration"]:
        print(f"\n{param}:")
        agg = df.groupby(param).agg({
            "ann_ret": "mean",
            "sharpe": "mean",
            "max_dd": "mean",
        }).round(3)
        print(agg.to_string())

    return df


def run_segmented_test(best_params):
    """分段稳定性测试."""
    con = duckdb.connect(DB, read_only=True)

    print("\n" + "=" * 70)
    print(f"分段稳定性测试: {best_params}")
    print("=" * 70)

    sig = build_signal_carry(
        con, variant="state_mom_curv2",
        z_window=best_params["z_window"],
        mom_window=best_params["mom_window"],
        min_duration=best_params["min_duration"],
    )
    r = backtest_carry(sig)
    nav = r["nav"]

    print(f"\n{'时段':<15} {'年化%':<10} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 50)

    for label, start, end in [
        ("2018-2020", "2018-01-01", "2020-12-31"),
        ("2021-2023", "2021-01-01", "2023-12-31"),
        ("2024-2025", "2024-01-01", "2025-12-31"),
        ("全样本", "2018-01-01", "2025-12-31"),
    ]:
        yn = nav[(nav.index >= start) & (nav.index <= end)]
        if len(yn) > 4:
            ann = yn.iloc[-1] / yn.iloc[0] - 1
            ann_ann = (1 + ann) ** (52 / len(yn)) - 1
            ret = yn.pct_change().dropna()
            vol = ret.std() * np.sqrt(52)
            sh = ann_ann / vol if vol > 0 else 0
            dd = (yn / yn.cummax() - 1).min()
            print(f"{label:<15} {ann_ann*100:<10.1f} {sh:<10.2f} {dd*100:<10.1f}")
        else:
            print(f"{label:<15} —")


def main():
    # 网格搜索
    df = run_grid_search()

    # 分析结果
    df = analyze_results(df)

    # 获取最优参数
    best_idx = df["sharpe"].idxmax()
    best_params = {
        "z_window": int(df.loc[best_idx, "z_window"]),
        "mom_window": int(df.loc[best_idx, "mom_window"]),
        "min_duration": int(df.loc[best_idx, "min_duration"]),
    }
    print(f"\n最优参数 (Sharpe最高): {best_params}")
    print(f"  Sharpe: {df.loc[best_idx, 'sharpe']:.3f}")
    print(f"  年化: {df.loc[best_idx, 'ann_ret']:.1f}%")
    print(f"  MaxDD: {df.loc[best_idx, 'max_dd']:.1f}%")

    # 分段验证
    run_segmented_test(best_params)

    # 保存结果
    df.to_csv("reports/cmf/carry_param_optimization.csv", index=False)
    print(f"\n结果已保存: reports/cmf/carry_param_optimization.csv")


if __name__ == "__main__":
    main()
