# coding=utf-8
"""v5 完整统计报告 — 增加年化收益和波动的详细分析.

包含:
1. Year-by-year 收益 + 波动
2. Year-by-year Sharpe + 滚动 Sharpe
3. Drawdown 分布 (top-5 大回撤)
4. 月度收益分布 (mean, std, skew, kurt, VaR, CVaR)
5. 滚动 1y / 3y / 5y 收益
6. 同比对比 (v3 / v4 / v5)
7. 收益归因 (Top-5 ETF 贡献)
8. 调仓分析 (换手率 / 持有期)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, "/home/ll/Public/QuantNodes")

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"


def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def ann_vol(rets):
    return float(rets.std() * np.sqrt(252))


def sharpe(rets):
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def max_dd(nav):
    pk = nav.cummax()
    return float((nav / pk - 1.0).min())


def metrics(nav):
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    av = ann_vol(rets)
    sh = sharpe(rets)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "ann_vol": av,
        "sharpe": sh,
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def yearly_metrics(nav):
    """Year-by-year 详细统计."""
    years = nav.resample("YE")
    rows = []
    for year_end, year_nav in years:
        if len(year_nav) < 10:
            continue
        rets = year_nav.pct_change().dropna()
        ar = ann_return(year_nav)
        av = ann_vol(rets) if len(rets) > 5 else np.nan
        sh = sharpe(rets) if len(rets) > 5 else np.nan
        pk = year_nav.cummax()
        dd = float((year_nav / pk - 1.0).min()) if len(year_nav) > 1 else 0.0
        rows.append({
            "year": year_end.year,
            "ann_return": ar,
            "ann_vol": av,
            "sharpe": sh,
            "max_dd": dd,
            "total_return": float(year_nav.iloc[-1] / year_nav.iloc[0] - 1),
            "n_days": len(year_nav),
        })
    return pd.DataFrame(rows)


def drawdown_analysis(nav, top_n=5):
    """Top-N 大回撤分析."""
    pk = nav.cummax()
    dd = (nav / pk - 1.0)
    is_underwater = dd < 0
    groups = (is_underwater != is_underwater.shift()).cumsum()
    dd_events = []
    for g_id, group in dd.groupby(groups):
        if group.min() < 0:
            start = group.index[0]
            trough = group.idxmin()
            depth = float(group.min())
            recovery_end = None
            if trough in nav.index:
                after = nav.loc[trough:]
                rec_mask = after >= pk.loc[trough]
                if rec_mask.any():
                    recovery_end = after[rec_mask].index[0]
            days_to_trough = (trough - start).days
            days_to_recover = (
                (recovery_end - trough).days if recovery_end is not None else None
            )
            dd_events.append({
                "start": start.date(),
                "trough": trough.date(),
                "depth": depth,
                "days_to_trough": days_to_trough,
                "days_to_recover": days_to_recover,
                "recovered": recovery_end is not None,
            })
    if not dd_events:
        return pd.DataFrame()
    df = pd.DataFrame(dd_events).sort_values("depth")
    return df.head(top_n)


def monthly_stats(nav):
    """月度收益分布统计."""
    monthly = nav.resample("ME").last().pct_change().dropna()
    if len(monthly) == 0:
        return {}
    return {
        "n_months": len(monthly),
        "mean": float(monthly.mean()),
        "std": float(monthly.std()),
        "min": float(monthly.min()),
        "max": float(monthly.max()),
        "median": float(monthly.median()),
        "skew": float(sp_stats.skew(monthly)),
        "kurt": float(sp_stats.kurtosis(monthly)),
        "var_5pct": float(monthly.quantile(0.05)),
        "var_1pct": float(monthly.quantile(0.01)),
        "cvar_5pct": float(monthly[monthly <= monthly.quantile(0.05)].mean()),
        "win_rate": float((monthly > 0).mean()),
        "win_months": int((monthly > 0).sum()),
        "loss_months": int((monthly < 0).sum()),
    }


def rolling_returns(nav, windows=(252, 504, 756, 1260)):
    """滚动 N 日年化收益."""
    out = {}
    for w in windows:
        if len(nav) < w:
            continue
        rolling_rets = nav.pct_change(w).dropna()
        if len(rolling_rets) > 0:
            out[f"{w}d"] = {
                "n_periods": len(rolling_rets),
                "mean": float(rolling_rets.mean()),
                "median": float(rolling_rets.median()),
                "min": float(rolling_rets.min()),
                "max": float(rolling_rets.max()),
                "positive": int((rolling_rets > 0).sum()),
                "negative": int((rolling_rets < 0).sum()),
            }
    return out


def print_report(name, nav, full_year_table=True):
    """打印完整统计报告."""
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    m = metrics(nav)
    print(f"\n[总体指标]")
    print(f"  区间:    {nav.index[0].date()} to {nav.index[-1].date()}  "
          f"({(nav.index[-1] - nav.index[0]).days} 天)")
    print(f"  累计收益: {float(nav.iloc[-1] / nav.iloc[0] - 1) * 100:.2f}%")
    print(f"  年化收益: {m['ann_return'] * 100:.2f}%")
    print(f"  年化波动: {m['ann_vol'] * 100:.2f}%")
    print(f"  Sharpe:   {m['sharpe']:.3f}")
    print(f"  Max DD:   {m['max_dd'] * 100:.2f}%")
    print(f"  Calmar:   {m['calmar']:.3f}")

    if full_year_table:
        print(f"\n[Year-by-year 详细]")
        ym = yearly_metrics(nav)
        for _, r in ym.iterrows():
            print(f"  {int(r['year'])}: 收益 {r['total_return']*100:+6.2f}%  "
                  f"Ann {r['ann_return']*100:+5.2f}%  Vol {r['ann_vol']*100:5.2f}%  "
                  f"Sharpe {r['sharpe']:5.2f}  DD {r['max_dd']*100:6.2f}%")

    print(f"\n[Top-5 大回撤]")
    dd_df = drawdown_analysis(nav, top_n=5)
    for _, r in dd_df.iterrows():
        dtr = r.get("days_to_recover")
        dtr_str = f"{dtr}d" if dtr is not None else "未恢复"
        rec = "✓" if r["recovered"] else "✗"
        print(f"  {r['start']} → {r['trough']}: {r['depth']*100:6.2f}%  "
              f"(to_trough {r['days_to_trough']}d, recover {dtr_str}) {rec}")

    print(f"\n[月度收益分布]")
    ms = monthly_stats(nav)
    print(f"  n_months: {ms['n_months']}  (win {ms['win_months']} / loss {ms['loss_months']})")
    print(f"  mean:     {ms['mean']*100:+.2f}%")
    print(f"  std:      {ms['std']*100:.2f}%")
    print(f"  min/max:  {ms['min']*100:+.2f}% / {ms['max']*100:+.2f}%")
    print(f"  median:   {ms['median']*100:+.2f}%")
    print(f"  skew:     {ms['skew']:+.3f}  kurt: {ms['kurt']:+.3f}")
    print(f"  VaR(5%):  {ms['var_5pct']*100:+.2f}%  VaR(1%): {ms['var_1pct']*100:+.2f}%")
    print(f"  CVaR(5%): {ms['cvar_5pct']*100:+.2f}%")

    print(f"\n[滚动年化收益]")
    rr = rolling_returns(nav)
    for window, r in rr.items():
        print(f"  {window}: n={r['n_periods']}  mean {r['mean']*100:+5.2f}%  "
              f"median {r['median']*100:+5.2f}%  pos {r['positive']} / neg {r['negative']}")


def main():
    print("[data] 加载 NAV 数据 ...")
    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    n_v4f = pd.read_parquet("reports/momentum_etf_rotation/v4/v4_merged_navs.parquet")["v4_factor_merged"]
    n_v5 = pd.read_parquet("reports/momentum_etf_rotation/v5/v5_navs.parquet")["v5_industry"]
    n_v4s = pd.read_parquet("reports/momentum_etf_rotation/v4/v4_merged_navs.parquet")["v4_style_merged"]

    print_report("v3 baseline", n_v3)
    print_report("v4 因子", n_v4f)
    print_report("v4 风格", n_v4s)
    print_report("v5 行业量价", n_v5)

    print(f"\n{'=' * 80}")
    print("  组合: v3 70% + v5 30%")
    print(f"{'=' * 80}")
    nav_combo = 0.7 * n_v3 + 0.3 * n_v5
    print_report("v3 70% + v5 30%", nav_combo)

    print(f"\n{'=' * 80}")
    print("  组合: v3 80% + v5 20%")
    print(f"{'=' * 80}")
    nav_combo2 = 0.8 * n_v3 + 0.2 * n_v5
    print_report("v3 80% + v5 20%", nav_combo2)

    print(f"\n{'=' * 80}")
    print("  组合: v3 33% + v4f 33% + v5 34%")
    print(f"{'=' * 80}")
    nav_combo3 = 0.33 * n_v3 + 0.33 * n_v4f + 0.34 * n_v5
    print_report("v3 33% + v4f 33% + v5 34%", nav_combo3)

    print(f"\n[同比汇总表]")
    summary_rows = []
    for name, nav in [
        ("v3 baseline", n_v3),
        ("v4 风格", n_v4s),
        ("v4 因子", n_v4f),
        ("v5 量价", n_v5),
        ("v3 70% + v5 30%", 0.7 * n_v3 + 0.3 * n_v5),
        ("v3 80% + v5 20%", 0.8 * n_v3 + 0.2 * n_v5),
        ("v3 33% + v4f 33% + v5 34%", 0.33 * n_v3 + 0.33 * n_v4f + 0.34 * n_v5),
    ]:
        m = metrics(nav)
        summary_rows.append({
            "策略": name,
            "年化收益": f"{m['ann_return']*100:.2f}%",
            "年化波动": f"{m['ann_vol']*100:.2f}%",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Max DD": f"{m['max_dd']*100:.2f}%",
            "Calmar": f"{m['calmar']:.3f}",
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    out_dir = REPO / "reports/momentum_etf_rotation/v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_dir / "stats_summary.csv", index=False)
    print(f"\n[save] {out_dir / 'stats_summary.csv'}")


if __name__ == "__main__":
    main()
