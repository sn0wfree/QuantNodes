# coding: utf-8
"""v7.6 Phase 6: β_path 断点分析.

目的: 用滚动 TV-PR 输出, 分析 β 系数时变结构.

用法:
   python3.11 scripts/v7_6_beta_stability.py

输出:
   reports/momentum_etf_rotation/v7_6_beta_stability.csv
   reports/momentum_etf_rotation/v7_6_beta_stability_report.md

输出字段:
   - |β[t] - β[t-1]| 各维度统计 (mean, max, std)
   - 断点频率 (TV 罚项内部)
   - β[t] 各维度 std (跨时间, 衡量时变性)
   - β[t] 在不同 regime 的稀疏性
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
}

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 6: β_path 断点分析")
    logging.info("=" * 60)

    logging.info("加载数据...")
    X_panel, Y, valid_codes = load_v7_6_data()
    T, N, K = X_panel.shape
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    cfg = V7_6Config(
        lambda_tv=DEFAULT_PARAMS["lambda_tv"],
        lambda_l1=DEFAULT_PARAMS["lambda_l1"],
        min_history=52,
        window_size=DEFAULT_PARAMS["window_size"],
    )

    # 估计 β_path
    logging.info("估计 β_path...")
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        window_size=cfg.window_size,
        rho=cfg.rho,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    beta_path = beta_path.values  # (T, K)
    logging.info("  β_path: %s, 时间=%d 周", beta_path.shape, beta_path.shape[0])

    # 1. |β[t] - β[t-1]| 分布
    diff_beta = np.diff(beta_path, axis=0)  # (T-1, K)
    abs_diff = np.abs(diff_beta)
    logging.info("  |Δβ| mean=%.4f, std=%.4f, max=%.4f",
                 abs_diff.mean(), abs_diff.std(), abs_diff.max())

    # 断点: |Δβ| > 0.05 (任意维度)
    breakpoints_per_step = (abs_diff > 0.05).any(axis=1)
    n_breakpoints = breakpoints_per_step.sum()
    bp_freq = n_breakpoints / len(abs_diff)
    logging.info("  断点频率: %d / %d = %.2f%%",
                 n_breakpoints, len(abs_diff), bp_freq * 100)

    # 2. β[t] 各维度 std (跨时间)
    beta_std_per_dim = np.nanstd(beta_path, axis=0)
    beta_mean_per_dim = np.nanmean(beta_path, axis=0)
    beta_cv_per_dim = np.where(
        np.abs(beta_mean_per_dim) > 1e-6,
        beta_std_per_dim / np.abs(beta_mean_per_dim),
        0
    )
    logging.info("  β 各维 std: min=%.4f, max=%.4f, mean=%.4f",
                 beta_std_per_dim.min(), beta_std_per_dim.max(),
                 beta_std_per_dim.mean())

    # 3. 维度间异质性
    beta_cv_max = beta_cv_per_dim.max()
    beta_cv_mean = beta_cv_per_dim.mean()
    logging.info("  β 各维 CV (std/|mean|): min=%.2f, max=%.2f, mean=%.2f",
                 beta_cv_per_dim.min(), beta_cv_max, beta_cv_mean)

    # 4. 时序相关性
    # β_path 的自相关系数
    acf_lag1 = np.array([
        np.corrcoef(beta_path[:-1, k], beta_path[1:, k])[0, 1]
        if not (np.isnan(beta_path[:, k]).any()) else np.nan
        for k in range(K)
    ])
    acf_mean = np.nanmean(acf_lag1)
    logging.info("  β[t] 与 β[t-1] 自相关: mean=%.4f", acf_mean)

    # 5. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "metric": "|Δβ| mean",
            "value": round(float(abs_diff.mean()), 4),
            "note": "β 相邻周绝对差均值",
        },
        {
            "metric": "|Δβ| std",
            "value": round(float(abs_diff.std()), 4),
            "note": "β 相邻周绝对差标准差",
        },
        {
            "metric": "|Δβ| max",
            "value": round(float(abs_diff.max()), 4),
            "note": "最大 β 单步变化",
        },
        {
            "metric": "n_breakpoints",
            "value": int(n_breakpoints),
            "note": f"断点频率 (|Δβ|>0.05)",
        },
        {
            "metric": "bp_freq",
            "value": round(float(bp_freq), 4),
            "note": "断点频率比",
        },
        {
            "metric": "beta_std_per_dim_mean",
            "value": round(float(beta_std_per_dim.mean()), 4),
            "note": "β 各维度 std 均值",
        },
        {
            "metric": "beta_cv_per_dim_max",
            "value": round(float(beta_cv_max), 4),
            "note": "β 各维 CV 最大值 (>1 即不稳定)",
        },
        {
            "metric": "beta_cv_per_dim_mean",
            "value": round(float(beta_cv_mean), 4),
            "note": "β 各维 CV 均值",
        },
        {
            "metric": "beta_acf_lag1_mean",
            "value": round(float(acf_mean), 4),
            "note": "β lag-1 自相关 (>0.7 即稳定)",
        },
    ]
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_beta_stability.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 分析
    print("\n" + "=" * 80)
    print("Phase 6 β_path 断点分析")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("判据")
    print("=" * 80)
    if bp_freq > 0.5:
        print(f"🔴 断点频率极高 ({bp_freq:.1%}) → β 估计不稳定")
    elif bp_freq > 0.3:
        print(f"🟡 断点频率较高 ({bp_freq:.1%})")
    else:
        print(f"🟢 断点频率合理 ({bp_freq:.1%})")

    if beta_cv_mean > 1.0:
        print(f"🔴 β 各维 CV 均值高 ({beta_cv_mean:.2f}) → 异质性大")
    elif beta_cv_mean > 0.5:
        print(f"🟡 β 各维 CV 均值 ({beta_cv_mean:.2f})")
    else:
        print(f"🟢 β 各维 CV 均值合理 ({beta_cv_mean:.2f})")

    if acf_mean < 0.5:
        print(f"🔴 β 时序自相关低 ({acf_mean:.2f}) → 噪声大")
    elif acf_mean < 0.7:
        print(f"🟡 β 时序自相关 ({acf_mean:.2f})")
    else:
        print(f"🟢 β 时序自相关高 ({acf_mean:.2f}) → 时变结构稳")

    return 0


if __name__ == "__main__":
    sys.exit(main())
