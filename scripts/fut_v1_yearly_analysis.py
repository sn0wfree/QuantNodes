#!/usr/bin/env python3
"""fut_v1 year-by-year NAV analysis: Sharpe, drawdown, rolling metrics, trend."""

import pandas as pd
import numpy as np

NAV_PATH = "output/fut_v1_72p_nav.parquet"
RISK_FREE = 0.02  # annualised
TRADING_DAYS = 252


def load_nav() -> pd.Series:
    df = pd.read_parquet(NAV_PATH)
    nav = df["nav"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    return nav


def year_metrics(nav: pd.Series, year: int) -> dict:
    mask = nav.index.year == year
    y = nav[mask]
    if len(y) < 10:
        return None
    ret = y.pct_change().dropna()
    ann_ret = ret.mean() * TRADING_DAYS
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0.0
    cum = y / y.iloc[0]
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    total_ret = y.iloc[-1] / y.iloc[0] - 1
    return {
        "Year": year,
        "Days": len(y),
        "Ann Return": f"{ann_ret:+.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Max DD": f"{max_dd:.1%}",
        "Calmar": f"{calmar:.2f}",
        "Total Ret": f"{total_ret:+.1%}",
    }


def rolling_12m_sharpe(nav: pd.Series) -> pd.Series:
    ret = nav.pct_change().dropna()
    roll_mean = ret.rolling(252).mean() * TRADING_DAYS
    roll_vol = ret.rolling(252).std() * np.sqrt(TRADING_DAYS)
    return (roll_mean - RISK_FREE) / roll_vol


def rolling_12m_maxdd(nav: pd.Series) -> pd.Series:
    result = pd.Series(index=nav.index, dtype=float)
    for i in range(252, len(nav)):
        window = nav.iloc[i - 252 : i + 1]
        cum = window / window.iloc[0]
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        result.iloc[i] = dd.min()
    return result


def main():
    nav = load_nav()
    years = sorted(nav.index.year.unique())

    # --- Year-by-year table ---
    rows = []
    for y in years:
        m = year_metrics(nav, y)
        if m:
            rows.append(m)
    df = pd.DataFrame(rows)
    print("=" * 90)
    print("FUT_V1 YEAR-BY-YEAR PERFORMANCE")
    print("=" * 90)
    print(df.to_string(index=False))

    # --- Best / worst ---
    sharpes = [float(r["Sharpe"]) for r in rows]
    best_idx = int(np.argmax(sharpes))
    worst_idx = int(np.argmin(sharpes))
    print(f"\nBest year:  {rows[best_idx]['Year']}  (Sharpe {rows[best_idx]['Sharpe']})")
    print(f"Worst year: {rows[worst_idx]['Year']}  (Sharpe {rows[worst_idx]['Sharpe']})")

    # --- Rolling metrics ---
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

    # --- Trend check ---
    print("\n" + "=" * 90)
    print("PERFORMANCE TREND ANALYSIS")
    print("=" * 90)
    sharpe_vals = np.array([float(r["Sharpe"]) for r in rows])
    year_vals = np.array([r["Year"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(year_vals, sharpe_vals, 1)
    print(f"  Sharpe linear trend slope: {slope:+.3f} per year")
    print(f"  {'DECLINING' if slope < -0.05 else 'STABLE' if abs(slope) < 0.05 else 'IMPROVING'} over time")

    # split halves
    mid = len(rows) // 2
    h1_sharpe = np.mean(sharpe_vals[:mid])
    h2_sharpe = np.mean(sharpe_vals[mid:])
    print(f"  First-half avg Sharpe:  {h1_sharpe:.2f}")
    print(f"  Second-half avg Sharpe: {h2_sharpe:.2f}")
    diff = h2_sharpe - h1_sharpe
    print(f"  Change: {diff:+.2f} ({'degradation' if diff < -0.1 else 'improvement' if diff > 0.1 else 'stable'})")

    # worst rolling windows
    print("\n  Worst rolling 12m drawdown periods:")
    sorted_dd = roll_dd.dropna().sort_values()
    for dt, val in sorted_dd.head(3).items():
        print(f"    {dt.strftime('%Y-%m-%d')}  {val:.1%}")


if __name__ == "__main__":
    main()
