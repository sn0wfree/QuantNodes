#!/usr/bin/env python3
"""scripts/cmf/test_inventory_cycle_atr_stop.py

测试四象限策略 + ATR止损
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_optimized,
    compute_inventory_cycle_with_atr_stop,
    backtest_inventory_cycle_optimized,
    backtest_inventory_cycle_with_atr_stop,
)

DB = Path("data/cmf/fundamental.duckdb")


def test_atr_stop():
    """测试ATR止损效果."""
    print("=" * 60)
    print("测试ATR止损效果")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 测试不同的ATR倍数
    atr_multiples = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    for atr_mult in atr_multiples:
        signal = compute_inventory_cycle_with_atr_stop(
            con=con,
            atr_multiple=atr_mult,
            atr_window=14,
        )
        result = backtest_inventory_cycle_with_atr_stop(signal)
        result["atr_multiple"] = atr_mult
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'ATR倍数':<10} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 40)

    for _, row in df.iterrows():
        print(f"{row['atr_multiple']:<10} {row['ann_ret']:<8.1%} {row['sharpe']:<8.2f} "
              f"{row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    # 最优ATR倍数
    best = df.iloc[0]
    print(f"\n最优ATR倍数: {best['atr_multiple']}")
    print(f"  年化: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MaxDD: {best['max_dd']:.1%}")

    return df


def compare_with_without_atr():
    """对比有无ATR止损."""
    print("\n" + "=" * 60)
    print("对比有无ATR止损")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 无ATR止损 (周频)
    signal_opt = compute_inventory_cycle_optimized(con=con)
    result_opt = backtest_inventory_cycle_optimized(signal_opt)

    # 有ATR止损 (日频)
    signal_atr = compute_inventory_cycle_with_atr_stop(
        con=con,
        atr_multiple=2.0,
        atr_window=14,
    )
    result_atr = backtest_inventory_cycle_with_atr_stop(signal_atr)

    print(f"\n{'策略':<20} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8} {'频率':<8}")
    print("-" * 60)
    print(f"{'无ATR止损':<20} {result_opt['ann_ret']:<8.1%} {result_opt['sharpe']:<8.2f} "
          f"{result_opt['max_dd']:<8.1%} {result_opt['win_rate']:<8.1%} {'周频':<8}")
    print(f"{'有ATR止损':<20} {result_atr['ann_ret']:<8.1%} {result_atr['sharpe']:<8.2f} "
          f"{result_atr['max_dd']:<8.1%} {result_atr['win_rate']:<8.1%} {'日频':<8}")

    return result_opt, result_atr


def test_overfitting():
    """过拟合检测."""
    print("\n" + "=" * 60)
    print("过拟合检测 (ATR止损)")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 全样本
    signal_full = compute_inventory_cycle_with_atr_stop(con=con, atr_multiple=2.0)
    result_full = backtest_inventory_cycle_with_atr_stop(signal_full)

    # IS (2014-2020)
    signal_is = signal_full[signal_full.index < pd.Timestamp("2021-01-01")]
    result_is = backtest_inventory_cycle_with_atr_stop(signal_is)

    # OOS (2021-2026)
    signal_oos = signal_full[signal_full.index >= pd.Timestamp("2021-01-01")]
    result_oos = backtest_inventory_cycle_with_atr_stop(signal_oos)

    print(f"\n{'期间':<15} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 45)
    print(f"{'全样本':<15} {result_full['ann_ret']:<8.1%} {result_full['sharpe']:<8.2f} "
          f"{result_full['max_dd']:<8.1%} {result_full['win_rate']:<8.1%}")
    print(f"{'IS (2014-2020)':<15} {result_is['ann_ret']:<8.1%} {result_is['sharpe']:<8.2f} "
          f"{result_is['max_dd']:<8.1%} {result_is['win_rate']:<8.1%}")
    print(f"{'OOS (2021-2026)':<15} {result_oos['ann_ret']:<8.1%} {result_oos['sharpe']:<8.2f} "
          f"{result_oos['max_dd']:<8.1%} {result_oos['win_rate']:<8.1%}")

    # 过拟合判断
    sharpe_ratio = result_oos["sharpe"] / result_is["sharpe"] if result_is["sharpe"] != 0 else 0
    print(f"\n过拟合判断:")
    print(f"  Sharpe OOS/IS = {sharpe_ratio:.2f}")

    if sharpe_ratio > 0.8:
        print(f"  ✓ 无过拟合 (OOS/IS > 0.8)")
    elif sharpe_ratio > 0.5:
        print(f"  ⚠ 轻度过拟合 (0.5 < OOS/IS < 0.8)")
    else:
        print(f"  ✗ 严重过拟合 (OOS/IS < 0.5)")

    return result_full, result_is, result_oos


if __name__ == "__main__":
    print("四象限策略 + ATR止损测试")
    print("=" * 60)

    # 测试ATR止损
    test_atr_stop()

    # 对比有无ATR止损
    compare_with_without_atr()

    # 过拟合检测
    test_overfitting()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
