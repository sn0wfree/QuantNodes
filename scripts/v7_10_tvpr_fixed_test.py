#!/usr/bin/env python3
# coding=utf-8
"""v7.10 TV-PR 验证: expanding-window (无 look-ahead).

三种方案对比:
  A. 原始: expanding(Y_orig), 组合用 Y_orig — 同周关系, beta 不含未来
  B. 修复: expanding(Y_fixed), 组合用 Y_orig — 预测关系, beta 不含未来
  C. full-sample (参考): tvpr_estimator(Y_orig), 组合用 Y_orig — 含未来函数

用法:
  python scripts/v7_10_tvpr_fixed_test.py
  python scripts/v7_10_tvpr_fixed_test.py --step 4
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
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
    tvpr_estimator,
)


def compute_metrics(nav: pd.Series, label: str) -> dict:
    ret = nav.pct_change().dropna()
    ann_ret = ret.mean() * 52
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    return {"label": label, "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar, "n_weeks": len(nav)}


def run_scenario(Y_train, X_train, Y_trade, X_trade, cfg, step, label):
    """运行一个场景: expanding-window TV-PR + construct_portfolio.

    Y_train: 用于训练 beta 的收益 (可能 shift 过)
    X_train: 用于训练 beta 的因子
    Y_trade: 用于 construct_portfolio 赚取收益的收益 (必须是原始 Y)
    X_trade: 用于 construct_portfolio 的因子
    """
    t0 = time.time()
    beta = expanding_window_tvpr(
        Y_train, X_train,
        lambda_tv=cfg.lambda_tv, lambda_l1=cfg.lambda_l1,
        min_history=52, rho=1.0, max_iter=200, tol=1e-5, step=step,
    )
    # expanding_window_tvpr 返回 (T, K) 的 beta
    # construct_portfolio 需要 DataFrame
    beta_df = pd.DataFrame(beta, index=Y_train.index)

    # 如果 Y_train 和 Y_trade 长度不同, 对齐 beta
    if len(Y_train) != len(Y_trade):
        # 补齐 beta 到 Y_trade 的长度
        extra = len(Y_trade) - len(Y_train)
        if extra > 0:
            beta_pad = np.tile(beta[-1:], (extra, 1))
            beta_full = np.vstack([beta, beta_pad])
        else:
            beta_full = beta[:len(Y_trade)]
        beta_df = pd.DataFrame(beta_full, index=Y_trade.index)

    nav, _ = construct_portfolio(Y_trade, X_trade, beta_df, cfg, return_weights=True)
    m = compute_metrics(nav, label)
    m["time"] = time.time() - t0
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=4, help="Beta 重估频率")
    args = parser.parse_args()

    print("=" * 70)
    print("v7.10 TV-PR 验证: expanding-window (无 look-ahead)")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/3] 加载 v7.10 数据...")
    X_panel, Y_orig, codes = load_v7_10_data()
    print(f"  X: {X_panel.shape}, Y: {Y_orig.shape}")

    # 2. 修复 Y
    print("\n[2/3] 修复 Y (shift -1)...")
    Y_fixed = Y_orig.shift(-1).iloc[:-1]
    X_fixed = X_panel[:-1]
    print(f"  X_fixed: {X_fixed.shape}, Y_fixed: {Y_fixed.shape}")

    # 3. 配置
    cfg = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, stop_loss_threshold=-0.15, stop_loss_cooldown=5)

    # 4. 三种场景
    print(f"\n[3/3] TV-PR 回测 (step={args.step})...")

    # A: 原始数据, expanding-window
    print("\n  --- A: expanding-window + 原始 Y (同周关系) ---")
    m_a = run_scenario(Y_orig, X_panel, Y_orig, X_panel, cfg, args.step, "A: 原始")
    print(f"  年化={m_a['ann_return']*100:+.2f}%, Sharpe={m_a['sharpe']:.3f}, DD={m_a['max_dd']*100:.2f}%, Calmar={m_a['calmar']:.3f} ({m_a['time']:.1f}s)")

    # B: 修复后数据, expanding-window, 组合用原始 Y
    print("\n  --- B: expanding-window + Y_fixed 训练, 原始 Y 组合 (预测关系) ---")
    m_b = run_scenario(Y_fixed, X_fixed, Y_orig, X_panel, cfg, args.step, "B: 修复后")
    print(f"  年化={m_b['ann_return']*100:+.2f}%, Sharpe={m_b['sharpe']:.3f}, DD={m_b['max_dd']*100:.2f}%, Calmar={m_b['calmar']:.3f} ({m_b['time']:.1f}s)")

    # C: full-sample (参考, 含未来函数)
    print("\n  --- C: full-sample TV-PR (含未来函数, 仅供对比) ---")
    t0 = time.time()
    beta_c = tvpr_estimator(Y_orig, X_panel, lambda_tv=cfg.lambda_tv, lambda_l1=cfg.lambda_l1,
                            method='admm', min_history=52, rho=1.0, max_iter=200, tol=1e-5)
    nav_c, _ = construct_portfolio(Y_orig, X_panel, beta_c, cfg, return_weights=True)
    m_c = compute_metrics(nav_c, "C: full-sample")
    m_c["time"] = time.time() - t0
    print(f"  年化={m_c['ann_return']*100:+.2f}%, Sharpe={m_c['sharpe']:.3f}, DD={m_c['max_dd']*100:.2f}%, Calmar={m_c['calmar']:.3f} ({m_c['time']:.1f}s)")

    # 5. 对比
    print("\n" + "=" * 70)
    print("对比结果")
    print("=" * 70)
    rows = [m_a, m_b, m_c]
    print(f"\n  {'场景':<20} {'年化':<10} {'Sharpe':<10} {'DD':<10} {'Calmar':<10}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['label']:<20} {r['ann_return']*100:+.2f}%{'':<4} {r['sharpe']:<10.3f} {r['max_dd']*100:.2f}%{'':<4} {r['calmar']:<10.3f}")

    # 6. 结论
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    print(f"\n  A (expanding + 原始 Y): Calmar {m_a['calmar']:.3f} — 同周关系 (beta 学的是 X[t] → Y[t] = t-1→t)")
    print(f"  B (expanding + Y_fixed): Calmar {m_b['calmar']:.3f} — 预测关系 (beta 学的是 X[t] → Y[t] = t→t+1)")
    print(f"  C (full-sample):         Calmar {m_c['calmar']:.3f} — 含未来函数 (TV penalty 耦合所有 beta)")

    drop_a_b = (m_a['calmar'] - m_b['calmar']) / m_a['calmar'] * 100 if m_a['calmar'] > 0 else 0
    drop_a_c = (m_a['calmar'] - m_c['calmar']) / m_a['calmar'] * 100 if m_a['calmar'] > 0 else 0

    print(f"\n  A→B 下降: {drop_a_b:.1f}% (去掉同周 look-ahead 的影响)")
    print(f"  A→C 下降: {drop_a_c:.1f}% (去掉 full-sample look-ahead 的影响)")

    if m_b['calmar'] > 0.5:
        print(f"\n  ✅ B (正确验证): Calmar {m_b['calmar']:.3f} > 0.5, 因子有真实预测信号")
    elif m_b['calmar'] > 0.2:
        print(f"\n  ⚠️ B (正确验证): Calmar {m_b['calmar']:.3f}, 信号较弱但真实")
    else:
        print(f"\n  ❌ B (正确验证): Calmar {m_b['calmar']:.3f}, 信号很弱, 需要新因子")


if __name__ == "__main__":
    main()
