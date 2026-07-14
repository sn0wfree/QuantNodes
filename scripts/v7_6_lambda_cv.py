# coding: utf-8
"""v7.6 λ 校验: Time Series CV 选择最优 λ_tv/λ_l1.

用法:
  python3.11 scripts/v7_6_lambda_cv.py

输出:
  reports/momentum_etf_rotation/v7_6_lambda_cv.csv
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# λ 候选范围
LAMBDA_TV_RANGE = [0.01, 0.05, 0.1, 0.5]
LAMBDA_L1_RANGE = [0.001, 0.01, 0.05, 0.1]

# Time Series CV 参数
N_SPLITS = 5
MIN_HISTORY = 12  # 最少 12 个月训练期


def compute_calmar(nav: pd.Series) -> float:
    """计算 Calmar."""
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / 12  # 月频
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    return ann_ret / abs(max_dd) if max_dd < 0 else 0.0


def time_series_cv_fold(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    train_end: int,
    val_end: int,
) -> float:
    """单折 Time Series CV.

    Args:
        Y: (T, N) 月频资产收益
        X_panel: (T, N, K) 月频因子值面板
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        train_end: 训练集结束索引
        val_end: 验证集结束索引

    Returns:
        calmar: 验证集 Calmar
    """
    # 训练集
    Y_train = Y.iloc[:train_end]
    X_train = X_panel[:train_end]

    # 验证集
    Y_val = Y.iloc[train_end:val_end]
    X_val = X_panel[train_end:val_end]

    if len(Y_train) < MIN_HISTORY or len(Y_val) < 3:
        return 0.0

    # 训练 TV-PR
    try:
        beta_path = tvpr_estimator(
            Y_train, X_train,
            lambda_tv=lambda_tv,
            lambda_l1=lambda_l1,
            min_history=MIN_HISTORY,
            max_iter=100,
            tol=1e-4,
        )
    except Exception as e:
        logging.warning("  TV-PR 训练失败: %s", e)
        return 0.0

    # 验证: 用最后一个 β_t 预测验证集
    beta_last = beta_path.iloc[-1].values  # (K,)

    # 用 X_val @ beta_last 得到资产分数
    # X_val is (T_val, N, K), beta_last is (K,)
    # scores = X_val[t] @ beta_last is (N,)
    nav_val = pd.Series(1.0, index=Y_val.index, dtype=float)
    for t in range(1, len(Y_val)):
        scores = X_val[t] @ beta_last  # (N,)
        top_n = min(10, len(scores))
        chosen_idx = np.argsort(scores)[-top_n:]
        chosen = Y_val.columns[chosen_idx]

        # 等权
        ret = Y_val[chosen].iloc[t].mean()
        nav_val.iloc[t] = nav_val.iloc[t - 1] * (1 + ret)

    return compute_calmar(nav_val)


def time_series_cv(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    n_splits: int = N_SPLITS,
) -> float:
    """Time Series CV 评估.

    Args:
        Y: (T, N) 月频资产收益
        X_panel: (T, N, K) 月频因子值面板
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        n_splits: 折数

    Returns:
        mean_calmar: 平均 Calmar
    """
    T = len(Y)
    fold_size = (T - MIN_HISTORY) // (n_splits + 1)

    if fold_size < 3:
        return 0.0

    scores = []
    for i in range(n_splits):
        train_end = MIN_HISTORY + (i + 1) * fold_size
        val_end = min(train_end + fold_size, T)

        calmar = time_series_cv_fold(Y, X_panel, lambda_tv, lambda_l1, train_end, val_end)
        scores.append(calmar)

    return np.mean(scores) if scores else 0.0


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 λ 校验: Time Series CV")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info("加载数据...")
    t0 = time.time()

    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
        load_monthly_macro_factors,
        load_monthly_pv_factors,
        load_monthly_asset_returns,
        build_mixed_factor_panel,
    )

    X_macro = load_monthly_macro_factors()
    X_pv = load_monthly_pv_factors()
    Y = load_monthly_asset_returns()

    # 构造面板
    asset_codes = list(Y.columns)
    X_panel, valid_codes = build_mixed_factor_panel(X_macro, X_pv, asset_codes)

    # 过滤有效资产
    Y = Y[valid_codes]

    t1 = time.time()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel.shape, Y.shape, t1 - t0)

    # 2. Grid Search
    logging.info("开始 Grid Search...")
    results = []

    for lambda_tv in LAMBDA_TV_RANGE:
        for lambda_l1 in LAMBDA_L1_RANGE:
            logging.info("  λ_tv=%.3f, λ_l1=%.3f ...", lambda_tv, lambda_l1)
            t0 = time.time()
            mean_calmar = time_series_cv(Y, X_panel, lambda_tv, lambda_l1)
            t1 = time.time()
            logging.info("    Calmar=%.4f, 耗时=%.1fs", mean_calmar, t1 - t0)
            results.append({
                "lambda_tv": lambda_tv,
                "lambda_l1": lambda_l1,
                "mean_calmar": round(mean_calmar, 4),
            })

    # 3. 找最优
    df = pd.DataFrame(results)
    best_idx = df["mean_calmar"].idxmax()
    best = df.iloc[best_idx]

    logging.info("=" * 60)
    logging.info("最优组合:")
    logging.info("  λ_tv: %.3f", best["lambda_tv"])
    logging.info("  λ_l1: %.3f", best["lambda_l1"])
    logging.info("  Mean Calmar: %.4f", best["mean_calmar"])
    logging.info("=" * 60)

    # 4. 保存结果
    output_dir = REPO / "reports/momentum_etf_rotation"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v7_6_lambda_cv.csv"
    df.to_csv(csv_path, index=False)
    logging.info("结果已保存: %s", csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
