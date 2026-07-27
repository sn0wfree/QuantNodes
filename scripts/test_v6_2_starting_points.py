# coding: utf-8
"""v6.2 起点依赖测试: 10 个起点, 计算 Calmar CV%.

用法:
  python3.11 scripts/test_v6_2_starting_points.py

输出:
  reports/momentum_etf_rotation/v6_2_starting_points.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import (
    V6_2Config,
    run_v6_2_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 10 个起点 (2018-01-01 ~ 2023-01-01 均匀分布)
START_POINTS = [
    "2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01",
    "2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01",
    "2022-01-01", "2023-01-01",
]

# 数据路径
CLOSE_PARQUET = REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet"
OHLCV_PARQUET = REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def calmar(nav: pd.Series) -> float:
    """计算 Calmar."""
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    return ann_ret / abs(max_dd) if max_dd < 0 else 0.0


def max_dd(nav: pd.Series) -> float:
    """计算最大回撤."""
    if nav.empty or len(nav) < 2:
        return 0.0
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    return float(dd.min())


def ann_return(nav: pd.Series) -> float:
    """计算年化收益."""
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    return float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)


def sharpe(nav: pd.Series) -> float:
    """计算夏普比率."""
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    return ann_ret / (rets.std() * np.sqrt(252))


def main() -> int:
    # 加载数据
    logging.info("加载数据...")
    panel_close = pd.read_parquet(CLOSE_PARQUET)
    panel_ohlcv = pd.read_parquet(OHLCV_PARQUET)
    logging.info("  panel_close: %s", panel_close.shape)
    logging.info("  panel_ohlcv: %s", panel_ohlcv.shape)

    # 对齐数据 (用 panel_close 的 index)
    common_idx = panel_close.index.intersection(panel_ohlcv.index)
    panel_close = panel_close.loc[common_idx]
    panel_ohlcv = panel_ohlcv.loc[common_idx]
    logging.info("  对齐后: %s", panel_close.shape)

    # 默认配置 (与 v6.2 PROMISING 一致)
    cfg = V6_2Config(
        use_orthogonal=True,
        sort_method="ir_expanding",
        min_history=120,
        ic_min_months=6,
    )

    results = []
    for start in START_POINTS:
        logging.info("测试起点: %s", start)

        # 切片数据
        mask = panel_close.index >= start
        pc = panel_close[mask]
        po = panel_ohlcv[mask]

        if len(pc) < 252:
            logging.warning("  数据不足 252 天, 跳过")
            continue

        # 跑回测
        try:
            nav = run_v6_2_backtest(pc, po, cfg)
            c = calmar(nav)
            results.append({
                "起点": start,
                "Calmar": round(c, 4),
                "最大回撤": round(max_dd(nav), 4),
                "年化": round(ann_return(nav), 4),
                "夏普": round(sharpe(nav), 4),
                "数据天数": len(nav),
            })
            logging.info("  Calmar=%.4f, MaxDD=%.4f, Ann=%.4f, Sharpe=%.4f",
                        c, max_dd(nav), ann_return(nav), sharpe(nav))
        except Exception as exc:
            logging.error("  失败: %s", exc)
            results.append({
                "起点": start,
                "Calmar": 0.0,
                "最大回撤": 0.0,
                "年化": 0.0,
                "夏普": 0.0,
                "数据天数": 0,
                "_error": str(exc),
            })

    # 计算统计
    df = pd.DataFrame(results)
    calmars = [r["Calmar"] for r in results if r["Calmar"] > 0]

    if len(calmars) >= 2:
        mean_c = np.mean(calmars)
        std_c = np.std(calmars)
        cv = std_c / mean_c if mean_c > 0 else 0.0
    else:
        mean_c = std_c = cv = 0.0

    # 输出
    logging.info("=" * 60)
    logging.info("起点依赖测试结果:")
    logging.info("  Mean Calmar: %.4f", mean_c)
    logging.info("  Std Calmar:  %.4f", std_c)
    logging.info("  CV%%:         %.1f%%", cv * 100)
    logging.info("  阈值:        25%%")
    logging.info("  结果:        %s", "PASS" if cv <= 0.25 else "FAIL")
    logging.info("=" * 60)

    # 保存 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "v6_2_starting_points.csv"
    df.to_csv(csv_path, index=False)
    logging.info("结果已保存: %s", csv_path)

    # 保存统计
    stats_path = OUTPUT_DIR / "v6_2_starting_points_stats.csv"
    stats_df = pd.DataFrame([{
        "mean_calmar": round(mean_c, 4),
        "std_calmar": round(std_c, 4),
        "cv_pct": round(cv * 100, 1),
        "threshold_pct": 25.0,
        "passed": cv <= 0.25,
    }])
    stats_df.to_csv(stats_path, index=False)
    logging.info("统计已保存: %s", stats_path)

    return 0 if cv <= 0.25 else 1


if __name__ == "__main__":
    sys.exit(main())
