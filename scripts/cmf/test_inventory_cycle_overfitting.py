#!/usr/bin/env python3
"""scripts/cmf/test_inventory_cycle_overfitting.py

检测库存周期四象限策略是否过拟合

方法:
1. 数据分割: IS (2014-2020) / OOS (2021-2026)
2. IS优化参数
3. OOS验证
4. 比较 IS vs OOS 表现
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
    analyze_quadrant_performance,
)

DB = Path("data/cmf/fundamental.duckdb")


def split_data(signal: pd.DataFrame, split_date: str = "2021-01-01"):
    """分割数据为IS和OOS."""
    split = pd.Timestamp(split_date)
    is_data = signal[signal.index < split]
    oos_data = signal[signal.index >= split]
    return is_data, oos_data


def optimize_on_is(is_data: pd.DataFrame):
    """在IS数据上优化参数."""
    print("=" * 60)
    print("IS数据优化")
    print("=" * 60)

    # 参数网格
    econ_windows = [13, 20, 26, 39, 52]
    inv_windows = [1, 4, 8, 13]

    results = []

    for econ_w in econ_windows:
        for inv_w in inv_windows:
            try:
                signal = compute_inventory_cycle_quadrant(
                    econ_window=econ_w,
                    inv_change_window=inv_w,
                )
                # 只用IS数据
                is_signal = signal[signal.index < pd.Timestamp("2021-01-01")]
                result = backtest_inventory_cycle(is_signal)
                result["econ_window"] = econ_w
                result["inv_window"] = inv_w
                results.append(result)
            except Exception as e:
                pass

    # 转换为DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\nIS数据最优参数 (2014-2020):")
    print(f"{'经济窗口':<8} {'库存窗口':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 50)

    for _, row in df.head(5).iterrows():
        print(f"{row['econ_window']:<8} {row['inv_window']:<8} {row['ann_ret']:<8.1%} "
              f"{row['sharpe']:<8.2f} {row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    return df


def test_on_oos(oos_data: pd.DataFrame, econ_window: int, inv_window: int):
    """在OOS数据上测试参数."""
    print(f"\n" + "=" * 60)
    print(f"OOS数据验证 (2021-2026, 经济窗口={econ_window}, 库存窗口={inv_window})")
    print("=" * 60)

    signal = compute_inventory_cycle_quadrant(
        econ_window=econ_window,
        inv_change_window=inv_window,
    )
    oos_signal = signal[signal.index >= pd.Timestamp("2021-01-01")]
    result = backtest_inventory_cycle(oos_signal)

    print(f"\nOOS表现:")
    print(f"  年化收益: {result['ann_ret']:.1%}")
    print(f"  年化波动: {result['ann_vol']:.1%}")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_dd']:.1%}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  周数: {result['n_weeks']}")

    return result


def compare_is_oos(is_result: dict, oos_result: dict):
    """比较IS和OOS表现."""
    print(f"\n" + "=" * 60)
    print("IS vs OOS 对比")
    print("=" * 60)

    print(f"\n{'指标':<12} {'IS':<10} {'OOS':<10} {'OOS/IS':<10}")
    print("-" * 40)

    metrics = [
        ("年化", "ann_ret"),
        ("Sharpe", "sharpe"),
        ("MaxDD", "max_dd"),
        ("胜率", "win_rate"),
    ]

    for name, key in metrics:
        is_val = is_result[key]
        oos_val = oos_result[key]
        ratio = oos_val / is_val if is_val != 0 else 0
        print(f"{name:<12} {is_val:<10.2f} {oos_val:<10.2f} {ratio:<10.2f}")

    # 过拟合判断
    print(f"\n过拟合判断:")
    sharpe_ratio = oos_result["sharpe"] / is_result["sharpe"] if is_result["sharpe"] != 0 else 0
    print(f"  Sharpe OOS/IS = {sharpe_ratio:.2f}")

    if sharpe_ratio > 0.8:
        print(f"  ✓ 无过拟合 (OOS/IS > 0.8)")
    elif sharpe_ratio > 0.5:
        print(f"  ⚠ 轻度过拟合 (0.5 < OOS/IS < 0.8)")
    else:
        print(f"  ✗ 严重过拟合 (OOS/IS < 0.5)")


def test_default_parameters():
    """测试默认参数在IS和OOS的表现."""
    print(f"\n" + "=" * 60)
    print("默认参数 (经济窗口=52, 库存窗口=13) IS vs OOS")
    print("=" * 60)

    signal = compute_inventory_cycle_quadrant()

    # 分割数据
    is_data, oos_data = split_data(signal)

    # IS表现
    is_result = backtest_inventory_cycle(is_data)
    print(f"\nIS (2014-2020):")
    print(f"  年化: {is_result['ann_ret']:.1%}")
    print(f"  Sharpe: {is_result['sharpe']:.2f}")
    print(f"  MaxDD: {is_result['max_dd']:.1%}")

    # OOS表现
    oos_result = backtest_inventory_cycle(oos_data)
    print(f"\nOOS (2021-2026):")
    print(f"  年化: {oos_result['ann_ret']:.1%}")
    print(f"  Sharpe: {oos_result['sharpe']:.2f}")
    print(f"  MaxDD: {oos_result['max_dd']:.1%}")

    # 对比
    compare_is_oos(is_result, oos_result)

    return is_result, oos_result


def main():
    """主函数."""
    print("库存周期四象限策略过拟合检测")
    print("=" * 60)

    # 1. IS优化
    signal = compute_inventory_cycle_quadrant()
    is_data, oos_data = split_data(signal)
    is_optimization = optimize_on_is(is_data)

    # 2. 最优参数在OOS测试
    best = is_optimization.iloc[0]
    best_econ = int(best["econ_window"])
    best_inv = int(best["inv_window"])

    is_best = is_optimization.iloc[0].to_dict()
    oos_best = test_on_oos(oos_data, best_econ, best_inv)

    # 3. IS vs OOS对比
    compare_is_oos(is_best, oos_best)

    # 4. 默认参数IS vs OOS
    test_default_parameters()

    # 5. 各象限OOS表现
    print(f"\n" + "=" * 60)
    print("各象限OOS表现 (2021-2026)")
    print("=" * 60)

    signal_oos = compute_inventory_cycle_quadrant()
    signal_oos = signal_oos[signal_oos.index >= pd.Timestamp("2021-01-01")]
    results_oos = analyze_quadrant_performance(signal_oos)

    print(f"\n{'象限':<30} {'周数':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 70)

    for q, stats in results_oos.items():
        print(f"{stats['name']:<30} {stats['n_weeks']:<8} {stats['ann_ret']:<8.1%} "
              f"{stats['sharpe']:<8.2f} {stats['max_dd']:<8.1%} {stats['win_rate']:<8.1%}")

    print(f"\n" + "=" * 60)
    print("过拟合检测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
