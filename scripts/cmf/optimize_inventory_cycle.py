#!/usr/bin/env python3
"""scripts/cmf/optimize_inventory_cycle.py

优化库存周期四象限参数

参数:
1. econ_window: 经济增长方向判断窗口 (周)
2. inv_change_window: 库存变化窗口 (周)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_quadrant,
    backtest_inventory_cycle,
)

DB = Path("data/cmf/fundamental.duckdb")


def optimize_parameters():
    """网格搜索优化参数."""
    print("=" * 60)
    print("库存周期四象限参数优化")
    print("=" * 60)

    # 参数网格
    econ_windows = [4, 8, 13, 20, 26, 39, 52]
    inv_windows = [1, 2, 4, 8, 13]

    results = []

    for econ_w in econ_windows:
        for inv_w in inv_windows:
            try:
                signal = compute_inventory_cycle_quadrant(
                    econ_window=econ_w,
                    inv_change_window=inv_w,
                )
                result = backtest_inventory_cycle(signal)
                result["econ_window"] = econ_w
                result["inv_window"] = inv_w
                results.append(result)
            except Exception as e:
                print(f"Error: econ_w={econ_w}, inv_w={inv_w}: {e}")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 按Sharpe排序
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'经济窗口':<8} {'库存窗口':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 50)

    for _, row in df.head(10).iterrows():
        print(f"{row['econ_window']:<8} {row['inv_window']:<8} {row['ann_ret']:<8.1%} "
              f"{row['sharpe']:<8.2f} {row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    # 最优参数
    best = df.iloc[0]
    print(f"\n最优参数:")
    print(f"  经济窗口: {best['econ_window']}周")
    print(f"  库存窗口: {best['inv_window']}周")
    print(f"  年化收益: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  最大回撤: {best['max_dd']:.1%}")
    print(f"  胜率: {best['win_rate']:.1%}")

    return df


def test_best_parameters():
    """测试最优参数的各象限表现."""
    print("\n" + "=" * 60)
    print("最优参数各象限表现")
    print("=" * 60)

    # 使用默认参数
    signal = compute_inventory_cycle_quadrant()

    # 统计各象限
    print(f"\n四象限分布:")
    for q in [1, 2, 3, 4]:
        count = (signal["quadrant"] == q).sum()
        pct = count / len(signal) * 100
        print(f"  象限{q}: {count}周 ({pct:.1f}%)")

    # 各象限表现
    from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
        analyze_quadrant_performance,
    )
    results = analyze_quadrant_performance(signal)

    print(f"\n{'象限':<30} {'周数':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 70)

    for q, stats in results.items():
        print(f"{stats['name']:<30} {stats['n_weeks']:<8} {stats['ann_ret']:<8.1%} "
              f"{stats['sharpe']:<8.2f} {stats['max_dd']:<8.1%} {stats['win_rate']:<8.1%}")

    return results


if __name__ == "__main__":
    optimize_parameters()
    test_best_parameters()
