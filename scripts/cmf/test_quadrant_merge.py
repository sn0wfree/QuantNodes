#!/usr/bin/env python3
"""scripts/cmf/test_quadrant_merge.py

测试方案B: 用四象限逻辑替代v6.2库存象限
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6, backtest_v6
from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_optimized,
    backtest_inventory_cycle_optimized,
)

DB = Path("data/cmf/fundamental.duckdb")


def get_v62_performance():
    """获取v6.2性能."""
    con = duckdb.connect(str(DB), read_only=True)
    signal = build_signal_v6('CU0', con)

    # 加载价格
    df = con.execute('SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date').df()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    close = df.set_index('trade_date')['close']

    result = backtest_v6(signal, close)
    return result, signal


def get_quadrant_performance():
    """获取四象限策略性能."""
    signal = compute_inventory_cycle_optimized()
    result = backtest_inventory_cycle_optimized(signal)
    return result, signal


def compare_signals():
    """对比v6.2和四象限的信号."""
    print("=" * 60)
    print("对比v6.2和四象限的信号")
    print("=" * 60)

    # 获取v6.2信号
    con = duckdb.connect(str(DB), read_only=True)
    v6_signal = build_signal_v6('CU0', con)

    # 获取四象限信号
    quad_signal = compute_inventory_cycle_optimized()

    # 对齐
    common = v6_signal.index.intersection(quad_signal.index)
    v6_pos = v6_signal.loc[common, 'position']
    quad_pos = quad_signal.loc[common, 'position']

    print(f"\n信号对比 (共同期间: {common[0]} ~ {common[-1]}):")
    print(f"  v6.2 仓位范围: {v6_pos.min():.2f} ~ {v6_pos.max():.2f}")
    print(f"  四象限 仓位范围: {quad_pos.min():.2f} ~ {quad_pos.max():.2f}")

    # 计算相关性
    corr = v6_pos.corr(quad_pos)
    print(f"  相关系数: {corr:.3f}")

    # 信号一致性
    v6_direction = (v6_pos > 0.5).astype(int)
    quad_direction = (quad_pos > 0.5).astype(int)
    agreement = (v6_direction == quad_direction).mean()
    print(f"  方向一致性: {agreement:.1%}")

    return v6_pos, quad_pos


def main():
    """主函数."""
    print("=" * 60)
    print("方案B测试: 用四象限逻辑替代v6.2库存象限")
    print("=" * 60)

    # 1. 获取v6.2性能
    print("\n[1] v6.2 性能:")
    v6_result, v6_signal = get_v62_performance()
    print(f"  年化: {v6_result['ann_ret']:.1%}")
    print(f"  Sharpe: {v6_result['sharpe']:.2f}")
    print(f"  MaxDD: {v6_result['max_dd']:.1%}")
    print(f"  胜率: {v6_result['win_rate']:.1%}")

    # 2. 获取四象限性能
    print("\n[2] 四象限 性能:")
    quad_result, quad_signal = get_quadrant_performance()
    print(f"  年化: {quad_result['ann_ret']:.1%}")
    print(f"  Sharpe: {quad_result['sharpe']:.2f}")
    print(f"  MaxDD: {quad_result['max_dd']:.1%}")
    print(f"  胜率: {quad_result['win_rate']:.1%}")

    # 3. 对比信号
    print("\n[3] 信号对比:")
    v6_pos, quad_pos = compare_signals()

    # 4. 测试组合策略
    print("\n[4] 组合策略测试:")
    con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df = con.execute('SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date').df()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    close_w = df.set_index('trade_date')['close'].resample('W-FRI').last()

    # 对齐信号
    common = v6_signal.index.intersection(quad_signal.index).intersection(close_w.index)
    v6_pos = v6_signal.loc[common, 'position'].fillna(0)
    quad_pos = quad_signal.loc[common, 'position'].fillna(0)
    close_w = close_w.reindex(common)

    # 组合1: v6.2 + 四象限 等权
    combo1_pos = (v6_pos * 0.5 + quad_pos * 0.5).clip(0, 1)

    # 组合2: v6.2 + 四象限 (v6.2为主)
    combo2_pos = (v6_pos * 0.7 + quad_pos * 0.3).clip(0, 1)

    # 组合3: v6.2 + 四象限 (四象限为主)
    combo3_pos = (v6_pos * 0.3 + quad_pos * 0.7).clip(0, 1)

    # 回测组合
    ret = close_w.pct_change()

    for name, pos in [("v6.2+四象限 50/50", combo1_pos),
                       ("v6.2+四象限 70/30", combo2_pos),
                       ("v6.2+四象限 30/70", combo3_pos)]:
        cost = pos.diff().abs() * 5 / 10000
        strat_ret = pos.shift(1) * ret - cost
        strat_ret = strat_ret.dropna()

        if len(strat_ret) == 0:
            continue

        nav = (1 + strat_ret).cumprod()
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        n_years = len(strat_ret) / 52
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = strat_ret.std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (nav / nav.cummax() - 1).min()
        win_rate = (strat_ret > 0).mean()

        print(f"\n  {name}:")
        print(f"    年化: {ann_ret:.1%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    MaxDD: {max_dd:.1%}")
        print(f"    胜率: {win_rate:.1%}")


if __name__ == "__main__":
    main()
