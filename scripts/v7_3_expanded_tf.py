# coding=utf-8
"""v7.3 + TF + 扩展 ETF 池回测.

v7.3 baseline (13 指数, 无 TF) → v7.4 expanded (56 ETF+指数) + 趋势过滤.
所有业绩指标使用 common/metrics.py.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "QuantNodes").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import (
    load_expanded_panel,
    load_factor_returns,
    load_benchmark_price,
    load_index_panel,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_3 import (
    V7_3Config,
    V7_4Config,
    run_v7_3_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

# ============================================================
# 配置
# ============================================================
START_DATES = [
    pd.Timestamp("2018-01-07"),
    pd.Timestamp("2020-01-06"),
    pd.Timestamp("2022-01-03"),
]


def print_metrics(name: str, nav: pd.Series):
    """使用 common/metrics.py 打印指标."""
    m = compute_metrics(nav, freq="W")
    print(f"  AnnRet:  {m['AnnRet']*100:+.2f}%")
    print(f"  Vol:     {m['Vol']*100:.2f}%")
    print(f"  Sharpe:  {m['Sharpe']:.3f}")
    print(f"  Sortino: {m['Sortino']:.3f}")
    print(f"  MaxDD:   {m['MaxDD']*100:.2f}%")
    print(f"  Calmar:  {m['Calmar']:.3f}")
    print(f"  WinRate: {m['WinRate']*100:.1f}%")
    return m


# ============================================================
# 1. v7.3 baseline (13 指数, 无 TF) — 对照组
# ============================================================
def run_v73_baseline():
    log = logging.getLogger("v7_3")
    log.info("v7.3 baseline: 13 indices, no TF")
    print("=" * 60)
    print("v7.3 baseline: 13 指数, 无 TF (对照组)")
    print("=" * 60)

    log.info("  loading index_panel...")
    index_panel = load_index_panel()
    log.info(f"  index_panel: {index_panel.shape}")

    log.info("  loading factor_returns...")
    factor_returns = load_factor_returns()
    log.info(f"  factor_returns: {factor_returns.shape}")

    cfg = V7_3Config(bootstrap_times=100)

    log.info("  running backtest...")
    nav = run_v7_3_backtest(index_panel, factor_returns, cfg)
    log.info(f"  backtest done, nav length: {len(nav)}")

    m_full = compute_metrics(nav, freq="W")
    m_oos = compute_metrics(nav, freq="W", oos_start="2022-01-01")

    print("\n=== Full Period ===")
    print_metrics("v7.3 baseline", nav)
    print("\n=== OOS 2022-2026 ===")
    for k, v in m_oos["OOS"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return m_full, m_oos["OOS"]


# ============================================================
# 2. v7.3 + TF (13 指数, 有 TF)
# ============================================================
def run_v73_tf():
    log = logging.getLogger("v7_3")
    log.info("v7.3 + TF: 13 indices, with TF")
    print("\n" + "=" * 60)
    print("v7.3 + TF: 13 指数, 趋势过滤")
    print("=" * 60)

    log.info("  loading index_panel...")
    index_panel = load_index_panel()
    log.info("  loading factor_returns...")
    factor_returns = load_factor_returns()
    log.info("  loading benchmark_price...")
    benchmark_price = load_benchmark_price("沪深300指数")
    log.info(f"  benchmark_price: {benchmark_price.shape}")

    cfg = V7_3Config(
        trend_filter_enabled=True,
        trend_filter_ma=200,
        trend_filter_bear=0.5,
        bootstrap_times=100,
    )

    log.info("  running backtest...")
    nav = run_v7_3_backtest(index_panel, factor_returns, cfg, benchmark_price=benchmark_price)
    log.info(f"  backtest done, nav length: {len(nav)}")

    m_full = compute_metrics(nav, freq="W")
    m_oos = compute_metrics(nav, freq="W", oos_start="2022-01-01")

    print("\n=== Full Period ===")
    print_metrics("v7.3 + TF", nav)
    print("\n=== OOS 2022-2026 ===")
    for k, v in m_oos["OOS"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return m_full, m_oos["OOS"]


# ============================================================
# 3. v7.3 expanded + TF (56 ETF+指数, 有 TF) — 主实验
# ============================================================
def run_v73_expanded_tf():
    log = logging.getLogger("v7_3")
    log.info("v7.3 expanded + TF: 56 ETF+indices, with TF")
    print("\n" + "=" * 60)
    print("v7.3 expanded + TF: 56 ETF+指数, 趋势过滤")
    print("=" * 60)

    log.info("  loading expanded_panel...")
    expanded_panel = load_expanded_panel()
    log.info(f"  expanded_panel: {expanded_panel.shape}")

    log.info("  loading factor_returns...")
    factor_returns = load_factor_returns()
    log.info("  loading benchmark_price...")
    benchmark_price = load_benchmark_price("沪深300指数")

    cfg = V7_4Config(
        trend_filter_enabled=True,
        trend_filter_ma=200,
        trend_filter_bear=0.5,
        bootstrap_times=100,
    )

    log.info("  running backtest (this may take a few minutes)...")
    nav = run_v7_3_backtest(expanded_panel, factor_returns, cfg, benchmark_price=benchmark_price)
    log.info(f"  backtest done, nav length: {len(nav)}")

    m_full = compute_metrics(nav, freq="W")
    m_oos = compute_metrics(nav, freq="W", oos_start="2022-01-01")

    print("\n=== Full Period ===")
    print_metrics("v7.3 expanded + TF", nav)
    print("\n=== OOS 2022-2026 ===")
    for k, v in m_oos["OOS"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return m_full, m_oos["OOS"]


# ============================================================
# 4. CV% 测试 (3 起点, v7.3 expanded + TF)
# ============================================================
def run_cv_test():
    log = logging.getLogger("v7_3")
    log.info("CV% test: 3 starting points")
    print("\n" + "=" * 60)
    print("v7.3 expanded + TF: CV% 测试 (3 起点)")
    print("=" * 60)

    log.info("  loading expanded_panel...")
    expanded_panel = load_expanded_panel()
    log.info("  loading factor_returns...")
    factor_returns = load_factor_returns()
    log.info("  loading benchmark_price...")
    benchmark_price = load_benchmark_price("沪深300指数")

    results = []
    for start_date in START_DATES:
        print(f"\n--- 起点: {start_date.strftime('%Y-%m-%d')} ---")

        # 截断数据
        mask_exp = expanded_panel.index >= start_date
        mask_fac = factor_returns.index >= start_date
        idx_panel = expanded_panel.loc[mask_exp]
        fac_panel = factor_returns.loc[mask_fac]

        # benchmark 也要截断
        bm = benchmark_price.loc[:start_date]
        # 但 TF 需要完整历史 (loc[:as_of] 在内部处理)

        cfg = V7_4Config(
            trend_filter_enabled=True,
            trend_filter_ma=200,
            trend_filter_bear=0.5,
            bootstrap_times=100,
        )

        try:
            nav = run_v7_3_backtest(idx_panel, fac_panel, cfg, benchmark_price=benchmark_price)
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
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": start_date.strftime("%Y-%m-%d"), "error": str(e)})

    # 汇总
    valid = [r for r in results if "error" not in r]
    if len(valid) < 2:
        print("⚠️ 有效起点不足 2 个")
        return

    calmars = [r["calmar"] for r in valid]
    calmar_mean = np.mean(calmars)
    calmar_std = np.std(calmars)
    calmar_cv = calmar_std / abs(calmar_mean) if abs(calmar_mean) > 1e-9 else float("inf")

    print(f"\n{'='*60}")
    print("CV% 汇总")
    print(f"{'='*60}")
    for r in valid:
        print(f"  {r['name']}: Calmar={r['calmar']:.3f}, Sharpe={r['sharpe']:.3f}, DD={r['max_dd']*100:.2f}%")
    print(f"\n  Calmar 均值: {calmar_mean:.3f}")
    print(f"  Calmar 标准差: {calmar_std:.3f}")
    print(f"  CV%: {calmar_cv*100:.1f}%")

    if calmar_mean < 0:
        verdict = "DEPRECATED"
    elif calmar_cv < 0.25:
        verdict = "PASS"
    elif calmar_cv < 0.50:
        verdict = "PROMISING"
    else:
        verdict = "DEPRECATED"
    print(f"  判定: {verdict}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("v7_3")

    log.info("=== START ===")

    log.info("Step 1/4: v7.3 baseline (13 indices, no TF)")
    run_v73_baseline()

    log.info("Step 2/4: v7.3 + TF (13 indices, with TF)")
    run_v73_tf()

    log.info("Step 3/4: v7.3 expanded + TF (56 ETF+indices, with TF)")
    run_v73_expanded_tf()

    log.info("Step 4/4: CV% test (3 starting points)")
    run_cv_test()

    log.info("=== DONE ===")
