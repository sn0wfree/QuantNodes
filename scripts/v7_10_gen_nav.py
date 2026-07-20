#!/usr/bin/env python3
"""只生成 v7.10 日频 NAV 并写入 parquet (不再重生成 HTML).

输出: reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet

用法:
    python3.10 scripts/v7_10_gen_nav.py

耗时: ~6 分钟 (Beta 估计 step=4)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio, calculate_daily_nav,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
MIN_HISTORY = 52
BEST_LAMBDA_TV = 0.06
BEST_LAMBDA_L1 = 0.105
PARQUET_PATH = REPO / "reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet"


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.10 日频 NAV 生成 (只生成, 不渲染 HTML)")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info("[1/4] 加载数据...")
    X, Y, codes = load_v7_10_data()
    daily_ret = load_daily_etf_returns()
    T, N, K = X.shape
    logging.info("  数据: X=%s, K=%d", X.shape, K)

    # 2. Beta 估计
    logging.info("[2/4] Beta 估计 (step=4)...")
    t0 = time.time()
    beta = expanding_window_tvpr(Y, X, BEST_LAMBDA_TV, BEST_LAMBDA_L1,
                                  min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4)
    logging.info("  Beta 估计耗时: %.1fs", time.time() - t0)

    # 3. 构造组合 + 日频 NAV
    logging.info("[3/4] 构造组合 + 日频 NAV...")
    t0 = time.time()
    cfg = V7_6Config()
    nav_w, weights_df = construct_portfolio(Y, X, beta, cfg, return_weights=True)
    nav_d = calculate_daily_nav(weights_df, daily_ret, cfg)
    nav_d = nav_d / nav_d.iloc[0]
    logging.info("  NAV 构造耗时: %.1fs", time.time() - t0)
    logging.info("  v7.10 日频 NAV: %d 天, %s ~ %s",
                 len(nav_d), nav_d.index[0].date(), nav_d.index[-1].date())

    # 4. 写入 parquet (替换 v7.6)
    logging.info("[4/4] 写入 parquet...")
    navs = pd.read_parquet(PARQUET_PATH)
    logging.info("  原 parquet: %s, 列=%s", navs.shape, navs.columns.tolist())
    if "v7.6 TV-PR" in navs.columns:
        navs = navs.drop(columns=["v7.6 TV-PR"])
    if "v7.10 TV-PR (标准化+CV)" not in navs.columns:
        navs["v7.10 TV-PR (标准化+CV)"] = nav_d.reindex(navs.index)
    else:
        navs["v7.10 TV-PR (标准化+CV)"] = nav_d.reindex(navs.index)
    navs.to_parquet(PARQUET_PATH)
    logging.info("  parquet 已更新: %s", PARQUET_PATH)

    logging.info("=" * 60)
    logging.info("完成! 现在可以运行 v7_10_regen_html.py 渲染 HTML (会复用缓存)")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())