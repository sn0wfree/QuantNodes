#!/usr/bin/env python3
# coding=utf-8
"""v7.10 TV-PR 验证: 修复 look-ahead 后重跑.

验证 TV-PR 是否过拟合:
  原始: Y[t] = t-1→t 收益 (同周 look-ahead)
  修复: Y[t] = t→t+1 收益 (shift -1, 正确预测对齐)

用法:
  python scripts/v7_10_tvpr_fixed_test.py
  python scripts/v7_10_tvpr_fixed_test.py --step 4  # 每4周重估
"""
from __future__ import annotations

import sys
import time
import argparse
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
    calculate_daily_nav,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator


def compute_metrics(nav: pd.Series, label: str) -> dict:
    """计算周频指标."""
    ret = nav.pct_change().dropna()
    ann_ret = ret.mean() * 52
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    return {
        "label": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "n_weeks": len(nav),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=4, help="Beta 重估频率")
    args = parser.parse_args()

    print("=" * 70)
    print("v7.10 TV-PR 验证: 修复 look-ahead 后重跑")
    print("=" * 70)

    # 1. 加载原始数据
    print("\n[1/4] 加载 v7.10 原始数据...")
    X_panel, Y_orig, codes = load_v7_10_data()
    print(f"  X: {X_panel.shape}, Y: {Y_orig.shape}, codes: {len(codes)}")
    print(f"  Y_orig[t=100, 0]: {Y_orig.iloc[100, 0]:.6f} (t-1→t 收益)")

    # 2. 修复 Y: shift -1
    print("\n[2/4] 修复 Y (shift -1)...")
    Y_fixed = Y_orig.shift(-1).iloc[:-1]
    X_fixed = X_panel[:-1]
    print(f"  X_fixed: {X_fixed.shape}, Y_fixed: {Y_fixed.shape}")
    print(f"  Y_fixed[t=100, 0]: {Y_fixed.iloc[100, 0]:.6f} (t→t+1 收益)")

    # 3. 验证因子相关性
    print("\n[3/4] 验证因子相关性...")
    from scipy.stats import spearmanr

    factor_names_csv = REPO / "data" / "high_freq_macro" / "v7_10_factor_names.csv"
    factor_names = factor_names_csv.read_text().strip().split("\n")[1:]

    T, N, K = X_fixed.shape
    print(f"  {'因子':<20} {'corr(X[t],Y_orig[t])':<20} {'corr(X[t],Y_fixed[t])':<20} {'修复效果':<10}")
    print(f"  {'-'*70}")

    for k in range(min(K, len(factor_names))):
        fname = factor_names[k]

        # 原始
        v1 = ~np.isnan(X_panel[:-1, :, k].ravel()) & ~np.isnan(Y_orig.iloc[:-1].values.ravel())
        c_orig = spearmanr(X_panel[:-1, :, k].ravel()[v1], Y_orig.iloc[:-1].values.ravel()[v1])[0] if v1.sum() > 100 else np.nan

        # 修复后
        v2 = ~np.isnan(X_fixed[:, :, k].ravel()) & ~np.isnan(Y_fixed.values.ravel())
        c_fixed = spearmanr(X_fixed[:, :, k].ravel()[v2], Y_fixed.values.ravel()[v2])[0] if v2.sum() > 100 else np.nan

        if abs(c_orig) > 0.3 or abs(c_fixed) > 0.3:
            effect = "✓ 修复" if abs(c_fixed) < abs(c_orig) * 0.5 else "⚠️ 仍高"
            print(f"  {fname:<20} {c_orig:<20.4f} {c_fixed:<20.4f} {effect}")

    # 4. 跑 TV-PR
    print(f"\n[4/4] TV-PR 回测 (step={args.step})...")

    # 4a. 原始数据 (baseline)
    print("\n  --- 原始数据 (Y[t] = t-1→t) ---")
    cfg = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, stop_loss_threshold=-0.15, stop_loss_cooldown=5)
    t0 = time.time()
    beta_orig = tvpr_estimator(
        Y_orig, X_panel,
        lambda_tv=cfg.lambda_tv, lambda_l1=cfg.lambda_l1,
        method='admm', min_history=52, rho=1.0, max_iter=200, tol=1e-5,
    )
    nav_orig, _ = construct_portfolio(Y_orig, X_panel, beta_orig, cfg, return_weights=True)
    m_orig = compute_metrics(nav_orig, "原始 (look-ahead)")
    print(f"  年化={m_orig['ann_return']*100:+.2f}%, Sharpe={m_orig['sharpe']:.3f}, DD={m_orig['max_dd']*100:.2f}%, Calmar={m_orig['calmar']:.3f} ({time.time()-t0:.1f}s)")

    # 4b. 修复后数据
    #    TV-PR 用 Y_fixed 训练 (学习 X[t] → Y[t] = t→t+1 收益 的预测关系)
    #    construct_portfolio 用原始 Y (正确时间对齐: 赚 Y[t+1] = t→t+1 收益)
    print("\n  --- 修复后数据 (beta 用 Y_fixed 训练, 组合用原始 Y) ---")
    t0 = time.time()
    beta_fixed = tvpr_estimator(
        Y_fixed, X_fixed,
        lambda_tv=cfg.lambda_tv, lambda_l1=cfg.lambda_l1,
        method='admm', min_history=52, rho=1.0, max_iter=200, tol=1e-5,
    )
    # beta_fixed 有 429 步, Y_orig 有 430 步
    # construct_portfolio 用 beta[t-1], 所以补齐最后一个 beta
    beta_fixed_full = np.vstack([beta_fixed, beta_fixed[-1:]])
    beta_fixed_df = pd.DataFrame(beta_fixed_full, index=Y_orig.index)
    nav_fixed, _ = construct_portfolio(Y_orig, X_panel, beta_fixed_df, cfg, return_weights=True)
    m_fixed = compute_metrics(nav_fixed, "修复后 (正确)")
    print(f"  年化={m_fixed['ann_return']*100:+.2f}%, Sharpe={m_fixed['sharpe']:.3f}, DD={m_fixed['max_dd']*100:.2f}%, Calmar={m_fixed['calmar']:.3f} ({time.time()-t0:.1f}s)")

    # 5. 对比
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    print(f"\n  {'指标':<14} {'原始 (look-ahead)':<20} {'修复后 (正确)':<20} {'变化':<10}")
    print(f"  {'-'*64}")
    print(f"  {'年化收益':<14} {m_orig['ann_return']*100:+.2f}%{'':<14} {m_fixed['ann_return']*100:+.2f}%{'':<14} {(m_fixed['ann_return']-m_orig['ann_return'])*100:+.2f}%")
    print(f"  {'Sharpe':<14} {m_orig['sharpe']:.3f}{'':<16} {m_fixed['sharpe']:.3f}{'':<16} {m_fixed['sharpe']-m_orig['sharpe']:+.3f}")
    print(f"  {'最大回撤':<14} {m_orig['max_dd']*100:.2f}%{'':<14} {m_fixed['max_dd']*100:.2f}%{'':<14} {(m_fixed['max_dd']-m_orig['max_dd'])*100:+.2f}%")
    print(f"  {'Calmar':<14} {m_orig['calmar']:.3f}{'':<16} {m_fixed['calmar']:.3f}{'':<16} {m_fixed['calmar']-m_orig['calmar']:+.3f}")

    # 6. 判断
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    calmar_drop = (m_orig['calmar'] - m_fixed['calmar']) / m_orig['calmar'] * 100 if m_orig['calmar'] > 0 else 0

    if m_fixed['calmar'] > 0.5:
        verdict = "✅ TV-PR 有效 (Calmar > 0.5), 因子有真实预测信号"
    elif m_fixed['calmar'] > 0.2:
        verdict = "⚠️ TV-PR 信号很弱 (Calmar 0.2-0.5), 部分过拟合"
    elif m_fixed['calmar'] > 0:
        verdict = "⚠️ TV-PR 几乎无效 (Calmar < 0.2), 严重过拟合"
    else:
        verdict = "❌ TV-PR 完全无效 (Calmar ≤ 0), 确认过拟合"

    print(f"\n  修复前 Calmar: {m_orig['calmar']:.3f}")
    print(f"  修复后 Calmar: {m_fixed['calmar']:.3f}")
    print(f"  下降幅度: {calmar_drop:.1f}%")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
