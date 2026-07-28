# coding=utf-8
"""v7.10 test2: Rolling Window Z-score 标准化测试.

与 standardize_v7_10 的区别:
  - 宏观因子: rolling(52) Z-score (替代 expanding Z-score)
  - k=11 (real_rate): rolling(156) rank 归一化 [0,1]
  - k=10,13,14,16: 不标准化

使用 common/metrics.py 计算指标, 每个起点独立加载原始数据 + 独立标准化.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "QuantNodes").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_9_data,
    standardize_v7_10_test2_correct,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

HF_DIR = ROOT / "data" / "high_freq_macro"


# ============================================================
# 1. 全期 OOS 回测
# ============================================================
def run_full_backtest():
    """全期回测 (2018-2026)."""
    print("=" * 60)
    print("v7.10 test2: Rolling Window Z-score 全期回测")
    print("=" * 60)

    X_raw, Y, codes = load_v7_9_data()
    factor_names = (HF_DIR / "v7_9_factor_names.csv").read_text().strip().split("\n")[1:]

    X = standardize_v7_10_test2_correct(X_raw, factor_names)
    print(f"X: {X.shape}, Y: {Y.shape}")

    cfg = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, method="expanding", min_history=52)

    beta = tvpr_estimator(Y, X, cfg.lambda_tv, cfg.lambda_l1,
                          method=cfg.method, min_history=cfg.min_history,
                          rho=cfg.rho, max_iter=cfg.max_iter, tol=cfg.tol)
    nav = construct_portfolio(Y, X, beta, cfg)

    m_full = compute_metrics(nav, freq="W")
    m_oos = compute_metrics(nav, freq="W", oos_start="2022-01-01")

    print("\n=== Full Period ===")
    for k, v in m_full.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\n=== OOS 2022-2026 ===")
    for k, v in m_oos["OOS"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return m_full, m_oos["OOS"]


# ============================================================
# 2. CV% 测试 (3 起点, 每起点独立标准化)
# ============================================================
START_DATES = [
    pd.Timestamp("2018-01-07"),
    pd.Timestamp("2020-01-06"),
    pd.Timestamp("2022-01-03"),
]


def run_cv_test():
    """3 起点 CV% 测试."""
    print("\n" + "=" * 60)
    print("v7.10 test2: Rolling Window Z-score CV% 测试")
    print("=" * 60)

    X_raw_full, Y_full, codes = load_v7_9_data()
    factor_names = (HF_DIR / "v7_9_factor_names.csv").read_text().strip().split("\n")[1:]

    results = []
    for start_date in START_DATES:
        print(f"\n--- 起点: {start_date.strftime('%Y-%m-%d')} ---")

        mask = np.array(Y_full.index >= start_date)
        Y = Y_full.loc[mask]
        idx = np.where(mask)[0]
        X_truncated = X_raw_full[idx]

        # 独立标准化
        X = standardize_v7_10_test2_correct(X_truncated, factor_names)
        print(f"  数据: {len(Y)} 周, 标准化后 NaN 比例: {np.isnan(X).mean():.4f}")

        cfg = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, method="expanding", min_history=52)
        beta = tvpr_estimator(Y, X, cfg.lambda_tv, cfg.lambda_l1,
                              method=cfg.method, min_history=cfg.min_history,
                              rho=cfg.rho, max_iter=cfg.max_iter, tol=cfg.tol)
        nav = construct_portfolio(Y, X, beta, cfg)
        m = compute_metrics(nav, freq="W")

        result = {
            "name": start_date.strftime("%Y-%m-%d"),
            "ann_return": m["AnnRet"],
            "sharpe": m["Sharpe"],
            "max_dd": m["MaxDD"],
            "calmar": m["Calmar"],
        }
        results.append(result)
        print(f"  Calmar={m['Calmar']:.3f}, Sharpe={m['Sharpe']:.3f}, DD={m['MaxDD']*100:.2f}%")

    # 汇总
    calmars = [r["calmar"] for r in results]
    calmar_mean = np.mean(calmars)
    calmar_std = np.std(calmars)
    calmar_cv = calmar_std / abs(calmar_mean) if abs(calmar_mean) > 1e-9 else float("inf")

    print(f"\n{'='*60}")
    print("CV% 汇总")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']}: Calmar={r['calmar']:.3f}, Sharpe={r['sharpe']:.3f}, DD={r['max_dd']*100:.2f}%")
    print(f"\n  Calmar 均值: {calmar_mean:.3f}")
    print(f"  Calmar 标准差: {calmar_std:.3f}")
    print(f"  CV%: {calmar_cv*100:.1f}%")

    if calmar_mean < 0:
        verdict = "DEPRECATED (所有起点 Calmar 为负)"
    elif calmar_cv < 0.25:
        verdict = "PASS"
    elif calmar_cv < 0.50:
        verdict = "PROMISING"
    else:
        verdict = "DEPRECATED (CV% > 50%)"
    print(f"  判定: {verdict}")

    return results, calmar_cv


if __name__ == "__main__":
    run_full_backtest()
    run_cv_test()
