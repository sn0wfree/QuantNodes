# coding: utf-8
"""v7.6 Phase 4: 缺失数据扰动.

目的: 通过随机 mask X 矩阵, 测试 v7.6 的抗噪能力.

用法:
   python3.11 scripts/v7_6_sensitivity_missing.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_missing.csv

实验设计:
   - rates = [0.05, 0.10, 0.20]
   - 每个 rate × 3 次 (random_state=42, 43, 44)
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
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    construct_portfolio, calculate_daily_nav, load_daily_etf_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252

# 缺失率 + 随机种子
RATES = [0.05, 0.10, 0.20]
SEEDS = [42, 43, 44]

# 冻结参数
DEFAULT_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
}

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def compute_metrics(nav: pd.Series, freq: int = DAYS_PER_YEAR) -> dict:
    """计算业绩指标."""
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    return {"calmar": round(calmar, 4), "ann_return": round(ann_ret, 4),
            "vol": round(vol, 4), "max_dd": round(max_dd, 4), "sharpe": round(sharpe, 4)}


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 4: 缺失数据扰动")
    logging.info("=" * 60)

    logging.info("加载数据...")
    X_panel, Y, valid_codes = load_v7_6_data()
    daily_returns_df = load_daily_etf_returns()
    T, N, K = X_panel.shape
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    cfg = V7_6Config(
        lambda_tv=DEFAULT_PARAMS["lambda_tv"],
        lambda_l1=DEFAULT_PARAMS["lambda_l1"],
        min_history=52,
        window_size=DEFAULT_PARAMS["window_size"],
    )

    # 1. 先跑 baseline (无 mask)
    logging.info("=" * 60)
    logging.info("[baseline] X 无 mask")
    t0 = time.time()
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
    nav_weekly, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)
    nav_daily = calculate_daily_nav(weights_df, daily_returns_df, cfg)
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    base_m = compute_metrics(nav_daily_oos)
    base_m["seconds"] = round(time.time() - t0, 1)
    base_m["rate"] = 0.0
    base_m["seed"] = -1
    base_m["status"] = "OK"
    base_m["note"] = "baseline"
    base_calmar = base_m["calmar"]
    rows = [base_m]
    logging.info("  baseline OOS Calmar=%.4f, %.1fs", base_calmar, base_m["seconds"])

    # 2. 各 rate + 各 seed 扰动
    for rate in RATES:
        for seed in SEEDS:
            logging.info("=" * 60)
            logging.info("[rate=%.2f, seed=%d] mask X (%.0f%% 缺失)",
                         rate, seed, rate * 100)

            # mask X 矩阵
            X_masked = X_panel.copy()
            rng = np.random.default_rng(seed)
            mask = rng.random(X_masked.shape) < rate
            X_masked[mask] = np.nan

            t0 = time.time()
            try:
                beta_path = tvpr_estimator(
                    Y, X_masked,
                    lambda_tv=cfg.lambda_tv,
                    lambda_l1=cfg.lambda_l1,
                    method=cfg.method,
                    min_history=cfg.min_history,
                    window_size=cfg.window_size,
                    rho=cfg.rho,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                )
                nav_weekly, weights_df = construct_portfolio(Y, X_masked, beta_path, cfg, return_weights=True)
                nav_daily = calculate_daily_nav(weights_df, daily_returns_df, cfg)
                nav_daily_oos = nav_daily.loc["2022-01-01":]
                m = compute_metrics(nav_daily_oos)
                m["seconds"] = round(time.time() - t0, 1)
                m["rate"] = rate
                m["seed"] = seed
                m["status"] = "OK"
                m["note"] = f"mask_{int(rate*100)}_{seed}"
                rows.append(m)
                degradation = (base_calmar - m["calmar"]) / base_calmar if base_calmar > 0 else 0
                logging.info("  Calmar=%.4f (退化 %.1f%%), %.1fs",
                             m["calmar"], degradation * 100, m["seconds"])
            except Exception as e:
                logging.error("  失败: %s", e)
                rows.append({"rate": rate, "seed": seed, "status": "FAIL",
                             "calmar": 0, "sharpe": 0, "ann_return": 0,
                             "vol": 0, "max_dd": 0, "seconds": 0,
                             "note": f"mask_{int(rate*100)}_{seed}"})

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_missing.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 分析
    print("\n" + "=" * 80)
    print("Phase 4 缺失扰动结果")
    print("=" * 80)
    print(df[["rate", "seed", "calmar", "sharpe", "max_dd", "status", "seconds"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"baseline OOS Calmar: {base_calmar:.4f}")
    print("=" * 80)
    for rate in RATES:
        sub = df[(df["rate"] == rate) & (df["status"] == "OK")]
        if len(sub) > 0:
            avg_calmar = sub["calmar"].mean()
            degradation = (base_calmar - avg_calmar) / base_calmar if base_calmar > 0 else 0
            print(f"{int(rate*100)}% 缺失: avg Calmar={avg_calmar:.4f}, "
                  f"平均退化={degradation * 100:+.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
