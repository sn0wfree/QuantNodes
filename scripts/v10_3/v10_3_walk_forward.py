"""v10.3 G (Trend-Adaptive) Nested Walk-Forward.

Phase 2.3: 验证 G scheme 在 purged/embargoed CV 下的稳定性.

折叠: train 4y, test 1y, step 1y
purge: 5d
embargo: 5d
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics  # noqa: E402
from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (  # noqa: E402
    scheme_g_trend_adaptive,
    scheme_f_conservative,
    scheme_d_signal_weighted,
    load_navs,
    compute_nav,
)

PURGE = 5
EMBARGO = 5


def make_folds(start: str, end: str, train_years=4, test_years=1, step_years=1):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    folds = []
    fold_start = start
    fid = 0
    while True:
        train_end = fold_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append({
            "id": fid,
            "label": f"F{fid:02d} {fold_start.year}-{test_end.year}",
            "train_start": fold_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_start += pd.DateOffset(years=step_years)
        fid += 1
    return folds


def evaluate_segment(prices: pd.DataFrame, scheme_fn, start: str, end: str, label: str = ""):
    """计算某个日期段的 Sharpe / Calmar / MaxDD."""
    sub = prices.loc[start:end]
    if len(sub) < 50:
        return None
    weights = scheme_fn(sub)
    nav = compute_nav(sub, weights)
    m = compute_metrics(nav.loc[start:end])
    return {
        "label": label,
        "Sharpe": m.get("Sharpe", 0),
        "Calmar": m.get("Calmar", 0),
        "AnnRet": m.get("AnnRet", 0),
        "MaxDD": m.get("MaxDD", 0),
        "n_days": len(nav.loc[start:end]),
    }


def main():
    prices = load_navs()
    print(f"Prices: {prices.index[0].date()} ~ {prices.index[-1].date()}, {len(prices)} days")

    folds = make_folds("2018-01-01", "2026-05-06")
    print(f"\n{len(folds)} folds: {[f['label'] for f in folds]}")

    # Walk-forward 评估
    results_g = []
    results_d = []
    results_f = []

    for f in folds:
        # 用整个 period 的 scheme 生成完整 NAV, 然后 slice 到 test 期
        nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices))
        nav_d = compute_nav(prices, scheme_d_signal_weighted(prices))
        nav_f = compute_nav(prices, scheme_f_conservative(prices))

        # 测试期 (purge/embargo)
        test_start = f["test_start"] + pd.Timedelta(days=PURGE)
        test_end = f["test_end"]

        for nav, label, results in [
            (nav_g, "G", results_g),
            (nav_d, "D", results_d),
            (nav_f, "F", results_f),
        ]:
            seg = nav.loc[test_start:test_end].dropna()
            if len(seg) < 50:
                continue
            m = compute_metrics(seg)
            results.append({
                "fold_id": f["id"],
                "fold_label": f["label"],
                "scheme": label,
                "Sharpe": m.get("Sharpe", 0),
                "Calmar": m.get("Calmar", 0),
                "AnnRet": m.get("AnnRet", 0),
                "MaxDD": m.get("MaxDD", 0),
                "n_days": len(seg),
            })

    df_g = pd.DataFrame(results_g)
    df_d = pd.DataFrame(results_d)
    df_f = pd.DataFrame(results_f)

    # 汇总
    print("\n" + "=" * 70)
    print("Nested Walk-Forward Results (purge=5d, embargo=5d)")
    print("=" * 70)
    print(f"\n{'Scheme':10s} {'Sharpe':>10s} {'Calmar':>10s} {'AnnRet':>10s} {'MaxDD':>10s}")
    print("-" * 55)
    for df, label in [(df_g, "G"), (df_d, "D"), (df_f, "F")]:
        if len(df) > 0:
            print(f"{label:10s} {df['Sharpe'].mean():>10.3f} {df['Calmar'].mean():>10.3f} "
                  f"{df['AnnRet'].mean()*100:>9.2f}% {df['MaxDD'].mean()*100:>9.2f}%")

    print("\nFold details:")
    print(f"{'Fold':10s} {'Scheme':8s} {'Sharpe':>8s} {'Calmar':>8s} {'AnnRet':>8s} {'MaxDD':>8s}")
    print("-" * 55)
    for df in [df_g, df_d, df_f]:
        for _, r in df.iterrows():
            print(f"{r['fold_label']:10s} {r['scheme']:8s} {r['Sharpe']:>8.3f} "
                  f"{r['Calmar']:>8.3f} {r['AnnRet']*100:>7.2f}% {r['MaxDD']*100:>7.2f}%")

    # 压力段验证
    print("\n" + "=" * 70)
    print("Stress Segments")
    print("=" * 70)
    stress_periods = [
        ("2022-01-01", "2022-12-31", "2022 熊市"),
        ("2023-01-01", "2023-12-31", "2023 震荡"),
        ("2024-01-01", "2024-12-31", "2024 多事件"),
        ("2025-01-01", "2026-05-06", "2025+ 近况"),
    ]
    for start, end, label in stress_periods:
        nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices))
        nav_d = compute_nav(prices, scheme_d_signal_weighted(prices))
        m_g = compute_metrics(nav_g.loc[start:end])
        m_d = compute_metrics(nav_d.loc[start:end])
        delta = m_g.get("Calmar", 0) - m_d.get("Calmar", 0)
        print(f"{label:15s}: D Calmar={m_d.get('Calmar', 0):>7.3f}, "
              f"G Calmar={m_g.get('Calmar', 0):>7.3f}, "
              f"Delta={delta:>+7.3f} {'✓' if delta > 0 else '✗'}")

    # 保存
    out_dir = ROOT / "reports" / "momentum_etf_rotation" / "v10_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([df_g, df_d, df_f]).to_csv(out_dir / "v10_3_walk_forward_results.csv", index=False)
    print(f"\n[保存] {out_dir / 'v10_3_walk_forward_results.csv'}")


if __name__ == "__main__":
    main()
