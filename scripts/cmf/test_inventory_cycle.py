#!/usr/bin/env python3
"""scripts/cmf/test_inventory_cycle.py

测试库存周期四象限策略

测试内容:
1. 四象限信号生成
2. 各象限表现分析
3. 与v2策略对比
4. T+1延迟测试
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_quadrant,
    compute_inventory_cycle_v2,
    backtest_inventory_cycle,
    analyze_quadrant_performance,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_momentum import (
    compute_inventory_momentum,
    backtest_inventory_momentum,
)

DB = Path("data/cmf/fundamental.duckdb")


def test_basic_quadrant():
    """测试基本四象限信号生成."""
    print("=" * 60)
    print("测试1: 基本四象限信号生成")
    print("=" * 60)

    signal = compute_inventory_cycle_quadrant()

    print(f"信号期间: {signal.index[0]} ~ {signal.index[-1]}")
    print(f"信号数量: {len(signal)}")
    print(f"\n四象限分布:")
    for q in [1, 2, 3, 4]:
        count = (signal["quadrant"] == q).sum()
        pct = count / len(signal) * 100
        print(f"  象限{q}: {count}周 ({pct:.1f}%)")

    print(f"\n仓位分布:")
    for pos_val in [1.0, 0.0, -1.0]:
        count = (signal["position"] == pos_val).sum()
        pct = count / len(signal) * 100
        print(f"  仓位={pos_val:.0f}: {count}周 ({pct:.1f}%)")

    return signal


def test_quadrant_performance():
    """测试各象限表现."""
    print("\n" + "=" * 60)
    print("测试2: 各象限表现分析")
    print("=" * 60)

    signal = compute_inventory_cycle_quadrant()
    results = analyze_quadrant_performance(signal)

    print(f"\n{'象限':<30} {'周数':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 70)

    for q, stats in results.items():
        print(f"{stats['name']:<30} {stats['n_weeks']:<8} {stats['ann_ret']:<8.1%} "
              f"{stats['sharpe']:<8.2f} {stats['max_dd']:<8.1%} {stats['win_rate']:<8.1%}")

    return results


def test_backtest():
    """测试回测结果."""
    print("\n" + "=" * 60)
    print("测试3: 回测结果")
    print("=" * 60)

    signal = compute_inventory_cycle_quadrant()
    result = backtest_inventory_cycle(signal)

    print(f"\n基本四象限策略:")
    print(f"  年化收益: {result['ann_ret']:.1%}")
    print(f"  年化波动: {result['ann_vol']:.1%}")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_dd']:.1%}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  周数: {result['n_weeks']}")

    return result


def test_v2_comparison():
    """与v2策略对比."""
    print("\n" + "=" * 60)
    print("测试4: 与v2策略对比")
    print("=" * 60)

    # 四象限策略
    signal_quad = compute_inventory_cycle_quadrant()
    result_quad = backtest_inventory_cycle(signal_quad)

    # v1策略
    con = duckdb.connect(str(DB), read_only=True)
    signal_v1 = compute_inventory_momentum(con)
    result_v1 = backtest_inventory_momentum(signal_v1)

    print(f"\n{'策略':<20} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 50)
    print(f"{'四象限':<20} {result_quad['ann_ret']:<8.1%} {result_quad['sharpe']:<8.2f} "
          f"{result_quad['max_dd']:<8.1%} {result_quad['win_rate']:<8.1%}")
    print(f"{'v1纯变化':<20} {result_v1['ann_ret']:<8.1%} {result_v1['sharpe']:<8.2f} "
          f"{result_v1['max_dd']:<8.1%} {result_v1['win_rate']:<8.1%}")

    return result_quad, result_v1


def test_v2_with_momentum():
    """测试带动量确认的四象限策略."""
    print("\n" + "=" * 60)
    print("测试5: 带动量确认的四象限策略")
    print("=" * 60)

    # 基本四象限
    signal_basic = compute_inventory_cycle_quadrant()
    result_basic = backtest_inventory_cycle(signal_basic)

    # 带动量确认的四象限
    signal_mom = compute_inventory_cycle_v2(use_momentum_confirm=True)
    result_mom = backtest_inventory_cycle(signal_mom)

    print(f"\n{'策略':<20} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 50)
    print(f"{'基本四象限':<20} {result_basic['ann_ret']:<8.1%} {result_basic['sharpe']:<8.2f} "
          f"{result_basic['max_dd']:<8.1%} {result_basic['win_rate']:<8.1%}")
    print(f"{'动量确认四象限':<20} {result_mom['ann_ret']:<8.1%} {result_mom['sharpe']:<8.2f} "
          f"{result_mom['max_dd']:<8.1%} {result_mom['win_rate']:<8.1%}")

    return result_basic, result_mom


def test_combination():
    """测试与v6.2+carry组合."""
    print("\n" + "=" * 60)
    print("测试6: 与v6.2+carry组合")
    print("=" * 60)

    # 加载v6.2
    from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6
    con = duckdb.connect(str(DB), read_only=True)

    # v6.2信号
    signal_v6 = build_signal_v6(con=con)
    pos_v6 = signal_v6["position"]

    # carry信号
    from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import build_signal_carry
    signal_carry = build_signal_carry(con)
    pos_carry = signal_carry["position"]

    # 四象限信号
    signal_quad = compute_inventory_cycle_quadrant()
    pos_quad = signal_quad["position"]

    # 对齐
    common = pos_v6.index.intersection(pos_carry.index).intersection(pos_quad.index)
    pos_v6 = pos_v6.reindex(common).fillna(0)
    pos_carry = pos_carry.reindex(common).fillna(0)
    pos_quad = pos_quad.reindex(common).fillna(0)

    # 组合: v6.2=50% + carry=30% + 四象限=20%
    pos_combo = (pos_v6 * 0.5 + pos_carry * 0.3 + pos_quad * 0.2).clip(0, 1)

    # 回测
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()
    close_w = close_w.reindex(common)

    ret = close_w.pct_change()
    strat_ret = pos_combo.shift(1) * ret
    cost = pos_combo.diff().abs() * 5 / 10000
    strat_ret = (strat_ret - cost).dropna()

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    print(f"\nv6.2(50%) + carry(30%) + 四象限(20%):")
    print(f"  年化收益: {ann_ret:.1%}")
    print(f"  年化波动: {ann_vol:.1%}")
    print(f"  Sharpe: {ann_ret / ann_vol:.2f}")
    print(f"  最大回撤: {max_dd:.1%}")

    return ann_ret, ann_vol, max_dd


if __name__ == "__main__":
    print("铜库存周期四象限策略测试")
    print("=" * 60)

    # 运行所有测试
    test_basic_quadrant()
    test_quadrant_performance()
    test_backtest()
    test_v2_comparison()
    test_v2_with_momentum()
    test_combination()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
