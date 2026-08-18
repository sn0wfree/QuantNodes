#!/usr/bin/env python3
"""fut_v1 year-by-year NAV analysis: Sharpe, drawdown, rolling metrics, trend."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import numpy as np
import pandas as pd

from scripts.common import (
    compute_metrics, run_strategy, yearly_breakdown, print_yearly,
    parse_args, get_products,
)


def rolling_12m_sharpe(nav: pd.Series) -> pd.Series:
    """滚动12个月Sharpe."""
    ret = nav.pct_change().dropna()
    roll_mean = ret.rolling(252).mean() * 252
    roll_vol = ret.rolling(252).std() * np.sqrt(252)
    return (roll_mean - 0.02) / roll_vol


def rolling_12m_maxdd(nav: pd.Series) -> pd.Series:
    """滚动12个月最大回撤."""
    result = pd.Series(index=nav.index, dtype=float)
    for i in range(252, len(nav)):
        window = nav.iloc[i - 252 : i + 1]
        cum = window / window.iloc[0]
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        result.iloc[i] = dd.min()
    return result


def main():
    args = parse_args("fut_v1 Year-by-Year Analysis")
    products = get_products(args.quick)
    n = len(products)

    print(f"Running strategy ({n} products)...")
    result, metrics = run_strategy(products=products, start=args.start, end=args.end)
    nav = result.nav

    # 年度绩效表
    print_yearly(nav)

    # 最佳/最差年份
    df = yearly_breakdown(nav)
    best_idx = df["Sharpe"].idxmax()
    worst_idx = df["Sharpe"].idxmin()
    print(f"\nBest year:  {int(df.loc[best_idx, 'Year'])}  (Sharpe {df.loc[best_idx, 'Sharpe']:.2f})")
    print(f"Worst year: {int(df.loc[worst_idx, 'Year'])}  (Sharpe {df.loc[worst_idx, 'Sharpe']:.2f})")

    # 滚动指标
    roll_sharpe = rolling_12m_sharpe(nav)
    roll_dd = rolling_12m_maxdd(nav)

    print("\n" + "=" * 90)
    print("ROLLING 12-MONTH SHARPE (latest 12 quarter-end snapshots)")
    print("=" * 90)
    quarterly = roll_sharpe.dropna().resample("Q-DEC").last().dropna()
    for dt, val in quarterly.tail(12).items():
        print(f"  {dt.strftime('%Y-%m')}  {val:.2f}")

    print("\n" + "=" * 90)
    print("ROLLING 12-MONTH MAX DRAWDOWN (latest 12 quarter-end snapshots)")
    print("=" * 90)
    quarterly_dd = roll_dd.dropna().resample("Q-DEC").last().dropna()
    for dt, val in quarterly_dd.tail(12).items():
        print(f"  {dt.strftime('%Y-%m')}  {val:.1%}")

    # 趋势分析
    print("\n" + "=" * 90)
    print("PERFORMANCE TREND ANALYSIS")
    print("=" * 90)
    sharpe_vals = df["Sharpe"].values
    year_vals = df["Year"].values.astype(float)
    slope, intercept = np.polyfit(year_vals, sharpe_vals, 1)
    print(f"  Sharpe linear trend slope: {slope:+.3f} per year")
    print(f"  {'DECLINING' if slope < -0.05 else 'STABLE' if abs(slope) < 0.05 else 'IMPROVING'} over time")

    mid = len(df) // 2
    h1_sharpe = sharpe_vals[:mid].mean()
    h2_sharpe = sharpe_vals[mid:].mean()
    print(f"  First-half avg Sharpe:  {h1_sharpe:.2f}")
    print(f"  Second-half avg Sharpe: {h2_sharpe:.2f}")
    diff = h2_sharpe - h1_sharpe
    print(f"  Change: {diff:+.2f} ({'degradation' if diff < -0.1 else 'improvement' if diff > 0.1 else 'stable'})")

    # 最差滚动窗口
    print("\n  Worst rolling 12m drawdown periods:")
    sorted_dd = roll_dd.dropna().sort_values()
    for dt, val in sorted_dd.head(3).items():
        print(f"    {dt.strftime('%Y-%m-%d')}  {val:.1%}")


if __name__ == "__main__":
    main()
