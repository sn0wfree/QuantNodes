# coding: utf-8
"""v7.6 Phase 3: Bootstrap 稳定性测试.

目的: 通过 Y 重采样, 评估 Calmar 在数据扰动下的方差.

用法:
   python3.11 scripts/v7_6_sensitivity_bootstrap.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_bootstrap.csv

实验设计:
   - 50 次 bootstrap (random_state=42..141)
   - 固定 λ 为当前值
   - 用 Y 重采样 (时间维度) 测试稳定性

注意: TV-PR 是时间序列回归, 不能简单 i.i.d. bootstrap.
   这里采用块 bootstrap (block_size=13 周 = 1 季度).
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

# Bootstrap 参数
N_BOOTSTRAP = 30  # 用 30 次 (避免太慢)
BLOCK_SIZE = 13   # 块大小 (1 季度)
SEEDS = list(range(42, 42 + N_BOOTSTRAP))

# 冻结参数
DEFAULT_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
}

OOS_START = "2022-01-01"
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


def block_bootstrap(Y: pd.DataFrame, X_panel: np.ndarray, block_size: int, seed: int):
    """块 bootstrap 重采样.

    Returns:
        Y_boot: 重采样后的 Y
        X_boot: 重采样后的 X_panel
    """
    rng = np.random.default_rng(seed)
    T = Y.shape[0]
    n_blocks = T // block_size + 1

    # 生成块序列 (随机打乱)
    blocks = []
    while sum(len(b) for b in blocks) < T:
        block_idx = rng.integers(0, T - block_size)
        blocks.append((block_idx, min(block_idx + block_size, T)))

    # 拼接块
    indices = []
    for start, end in blocks:
        indices.extend(range(start, end))
        if len(indices) >= T:
            break

    indices = indices[:T]
    Y_boot = Y.iloc[indices].reset_index(drop=True)
    Y_boot.index = Y.index  # 保持原时间索引
    X_boot = X_panel[indices]
    return Y_boot, X_boot


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 3: Bootstrap 稳定性测试")
    logging.info("=" * 60)

    logging.info("加载数据...")
    t0 = time.time()
    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel.shape, Y.shape, time.time() - t0)

    cfg = V7_6Config(
        lambda_tv=DEFAULT_PARAMS["lambda_tv"],
        lambda_l1=DEFAULT_PARAMS["lambda_l1"],
        min_history=52,
        window_size=DEFAULT_PARAMS["window_size"],
    )

    # 加载日频数据 (用于日频 NAV)
    daily_returns_df = load_daily_etf_returns()

    rows = []
    for i, seed in enumerate(SEEDS):
        logging.info("=" * 60)
        logging.info("[bootstrap %d/%d] seed=%d", i + 1, N_BOOTSTRAP, seed)

        # 块重采样
        t1 = time.time()
        Y_boot, X_boot = block_bootstrap(Y, X_panel, BLOCK_SIZE, seed)
        sampling_time = time.time() - t1

        # TV-PR 估计
        t2 = time.time()
        try:
            beta_path = tvpr_estimator(
                Y_boot, X_boot,
                lambda_tv=cfg.lambda_tv,
                lambda_l1=cfg.lambda_l1,
                method=cfg.method,
                min_history=cfg.min_history,
                window_size=cfg.window_size,
                rho=cfg.rho,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
            )
            est_time = time.time() - t2
        except Exception as e:
            logging.error("  TV-PR 估计失败: %s", e)
            rows.append({"seed": seed, "status": "FAIL_TVPR", "calmar": 0,
                         "sharpe": 0, "ann_return": 0, "vol": 0, "max_dd": 0,
                         "sampling_seconds": round(sampling_time, 2)})
            continue

        # 构造组合 (周频 NAV)
        t3 = time.time()
        try:
            nav_weekly, weights_df = construct_portfolio(
                Y_boot, X_boot, beta_path, cfg, return_weights=True
            )
            # 日频 NAV
            nav_daily = calculate_daily_nav(weights_df, daily_returns_df, cfg)
            port_time = time.time() - t3
        except Exception as e:
            logging.error("  组合构造失败: %s", e)
            rows.append({"seed": seed, "status": "FAIL_PORT", "calmar": 0,
                         "sharpe": 0, "ann_return": 0, "vol": 0, "max_dd": 0,
                         "sampling_seconds": round(sampling_time, 2)})
            continue

        # 只看 OOS 段 (2022+) - 用 daily_returns 日历找到对应区间
        # 由于 bootstrap 重采样后索引已被替换, 我们看权重对应的调仓日
        rebal_dates = sorted(weights_df["date"].unique())
        # OOS 段 = OOS_START 之后的调仓日
        oos_mask = pd.to_datetime(rebal_dates) >= pd.Timestamp(OOS_START)
        if oos_mask.sum() == 0:
            oos_nav = nav_daily.iloc[-252 * 2:]  # 退回到最后 2 年
        else:
            first_oos_rebal = pd.Timestamp(rebal_dates[oos_mask.values.argmax()])
            # 找到 daily_returns 中 ≥ first_oos_rebal 的最后
            valid_dates = daily_returns_df.index[daily_returns_df.index >= first_oos_rebal]
            if len(valid_dates) == 0:
                continue
            oos_start_date = valid_dates[0]
            oos_nav = nav_daily.loc[oos_start_date:]

        m = compute_metrics(oos_nav)
        m["seed"] = seed
        m["status"] = "OK"
        m["sampling_seconds"] = round(sampling_time, 2)
        m["est_seconds"] = round(est_time, 2)
        m["port_seconds"] = round(port_time, 2)
        rows.append(m)
        logging.info("  Calmar=%.4f, Sharpe=%.2f, 年化=%.2f%%, DD=%.2f%%",
                     m["calmar"], m["sharpe"], m["ann_return"] * 100, m["max_dd"] * 100)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_bootstrap.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 分析
    ok = df[df["status"] == "OK"]
    if len(ok) >= 5:
        cals = ok["calmar"].astype(float)
        sharpes = ok["sharpe"].astype(float)
        cvs = cals.std() / cals.mean() if cals.mean() > 0 else 0

        print("\n" + "=" * 80)
        print(f"Bootstrap N={len(ok)}, 块大小={BLOCK_SIZE} 周")
        print("=" * 80)
        print(f"Calmar: mean={cals.mean():.4f}, std={cals.std():.4f}, CV={cvs:.4f}")
        print(f"Sharpe: mean={sharpes.mean():.4f}, std={sharpes.std():.4f}")
        print(f"Calmar 分位数: p5={cals.quantile(0.05):.4f}, "
              f"p50={cals.quantile(0.5):.4f}, p95={cals.quantile(0.95):.4f}")

        print("\n判据:")
        if cvs > 0.5:
            print(f"🔴 Bootstrap 高度不稳定 (CV={cvs:.2%})")
        elif cvs > 0.3:
            print(f"🟡 Bootstrap 中度不稳定 (CV={cvs:.2%})")
        else:
            print(f"🟢 Bootstrap 稳定 (CV={cvs:.2%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
