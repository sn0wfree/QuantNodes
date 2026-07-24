#!/usr/bin/env python3
# coding=utf-8
"""v7.12 DCC 综合测试: Plan A (因子池) + Plan B (regime overlay).

用法:
  python scripts/v7_12_dcc_test.py --full
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data,
    load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
from QuantNodes.strategy.momentum_etf_rotation.v7.enhanced_factors_v7_11 import compute_dcc_features
from QuantNodes.strategy.momentum_etf_rotation.v7.dcc_regime_overlay import DCCRegimeOverlay


def run_scenario(X, Y, cfg, step, label, dcc_overlay=None, dcc_scores=None):
    """运行一个场景."""
    t0 = time.time()
    Y_shifted = Y.shift(-1).iloc[:-1]
    X_shifted = X[:-1]
    beta = expanding_window_tvpr(Y_shifted, X_shifted, cfg.lambda_tv, cfg.lambda_l1,
                                  min_history=52, step=step)
    extra = len(Y) - len(Y_shifted)
    beta_full = np.vstack([beta, np.tile(beta[-1:], (extra, 1))]) if extra > 0 else beta
    beta_df = pd.DataFrame(beta_full, index=Y.index)
    nav, wdf = construct_portfolio(Y, X, beta_df, cfg, return_weights=True)

    # 应用 DCC overlay
    if dcc_overlay is not None and dcc_scores is not None:
        nav = dcc_overlay.apply(nav, wdf, dcc_scores)

    ret = nav.pct_change().dropna()
    ar = ret.mean() * 52
    av = ret.std() * np.sqrt(52)
    sh = ar / av if av > 0 else 0
    pk = nav.cummax()
    dd = (nav - pk) / pk
    mdd = dd.min()
    cal = ar / abs(mdd) if abs(mdd) > 0 else 0
    elapsed = time.time() - t0

    return {"label": label, "calmar": cal, "sharpe": sh, "ann_ret": ar, "max_dd": mdd, "time": elapsed}


def main():
    print("=" * 70)
    print("v7.12 DCC 综合测试: Plan A (因子池) + Plan B (regime overlay)")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    X_v710, Y, codes = load_v7_10_data()
    X_v712 = np.load(REPO / "data" / "high_freq_macro" / "v7_12_X_panel.npy")

    # 计算 DCC 分数 (用于 overlay)
    daily_returns = load_daily_etf_returns()
    daily_returns = daily_returns[[c for c in codes if c in daily_returns.columns]]
    dcc_daily = compute_dcc_features(daily_returns, window=60, min_periods=30)
    dcc_weekly = dcc_daily.resample("W-FRI").last()

    # 对齐 DCC 分数到 Y.index
    dcc_scores = pd.Series(np.nan, index=Y.index)
    for i, target_date in enumerate(Y.index):
        diffs = abs(dcc_weekly.index - target_date)
        if len(diffs) > 0:
            closest_idx = diffs.argmin()
            if diffs[closest_idx].days <= 7:
                dcc_scores.iloc[i] = dcc_weekly["dcc_zscore_mean"].iloc[closest_idx]

    print(f"  v7.10: {X_v710.shape}, v7.12: {X_v712.shape}")
    print(f"  DCC zscore_mean: mean={dcc_scores.mean():.3f}, max={dcc_scores.max():.3f}")

    cfg = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, stop_loss_threshold=-0.15, stop_loss_cooldown=5)
    step = 40  # 加速

    # 测试场景
    results = []

    print(f"\n{'='*70}")
    print("Plan A: DCC 加入因子池")
    print(f"{'='*70}")
    results.append(run_scenario(X_v710, Y, cfg, step, "v7.10 (36 因子)"))
    results.append(run_scenario(X_v712, Y, cfg, step, "v7.12 (36+6 DCC)"))

    print(f"\n{'='*70}")
    print("Plan B: DCC regime overlay (不同阈值)")
    print(f"{'='*70}")

    for threshold in [1.0, 1.5, 2.0]:
        for mode, factor in [("reduce", 0.5), ("cash", 0.0)]:
            label = f"DCC>{threshold} {mode}" + (f"({factor})" if mode == "reduce" else "")
            overlay = DCCRegimeOverlay(threshold=threshold, defense_mode=mode, reduce_factor=factor, cooldown=4)
            results.append(run_scenario(X_v710, Y, cfg, step, label, dcc_overlay=overlay, dcc_scores=dcc_scores))

    # 汇总
    print(f"\n{'='*70}")
    print("汇总结果")
    print(f"{'='*70}")
    print(f"\n  {'场景':<25} {'Calmar':<10} {'Sharpe':<10} {'年化':<10} {'DD':<10}")
    print(f"  {'-'*65}")

    baseline_cal = results[0]["calmar"]
    for r in results:
        delta = (r["calmar"] / baseline_cal - 1) * 100 if baseline_cal > 0 else 0
        marker = "✓" if r["calmar"] > baseline_cal * 1.05 else ""
        print(f"  {r['label']:<25} {r['calmar']:<10.3f} {r['sharpe']:<10.3f} {r['ann_ret']*100:+.2f}%{'':<4} {r['max_dd']*100:.2f}%  {marker}")

    # 最优
    best = max(results, key=lambda x: x["calmar"])
    print(f"\n  最优: {best['label']}, Calmar={best['calmar']:.3f}")
    if best["calmar"] > baseline_cal:
        print(f"  改善: {baseline_cal:.3f} → {best['calmar']:.3f} ({(best['calmar']/baseline_cal-1)*100:+.1f}%)")
    else:
        print(f"  无改善 (baseline={baseline_cal:.3f})")


if __name__ == "__main__":
    main()
