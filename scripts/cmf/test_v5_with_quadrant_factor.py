#!/usr/bin/env python3
"""scripts/cmf/test_v5_with_quadrant_factor.py

测试方案A: 将四象限信号作为v5的因子
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
from QuantNodes.strategy.commodity_fundamental.sign_constrained_lasso import rolling_zscore

DB = Path("data/cmf/fundamental.duckdb")


def compute_quadrant_signal(con=None, econ_window=52, inv_window=13):
    """计算四象限信号 (二元: 0或1)."""
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 加载库存
    df_inv = con.execute(
        "SELECT trade_date, value AS inventory "
        "FROM inventory_daily "
        "WHERE value_col='铜-库存' "
        "ORDER BY trade_date"
    ).df()
    df_inv["trade_date"] = pd.to_datetime(df_inv["trade_date"])
    inv_w = df_inv.set_index("trade_date")["inventory"].resample("W-FRI").last()

    # 对齐
    common = close_w.index.intersection(inv_w.index)
    close_w = close_w.reindex(common)
    inv_w = inv_w.reindex(common)

    # 经济增长方向: 铜价52周动量 > 0
    econ_momentum = close_w.pct_change(econ_window)
    econ_up = econ_momentum > 0

    # 库存变化方向: 库存13周变化率 > 0
    inv_change = inv_w.pct_change(inv_window)
    inv_up = inv_change > 0

    # 四象限信号: 被动去库或主动补库时为1，否则为0
    quadrant_signal = pd.Series(0.0, index=common)
    quadrant_signal[econ_up & ~inv_up] = 1.0  # 被动去库
    quadrant_signal[econ_up & inv_up] = 1.0   # 主动补库
    # quadrant_signal[~econ_up & inv_up] = 0.0  # 被动补库
    # quadrant_signal[~econ_up & ~inv_up] = 0.0  # 主动去库

    return quadrant_signal


def test_v5_with_quadrant_factor():
    """测试v5 + 四象限因子."""
    print("=" * 60)
    print("方案A测试: 将四象限信号作为v5的因子")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 1. 获取原始v6.2性能
    print("\n[1] 原始v6.2 性能:")
    signal_v6 = build_signal_v6('CU0', con)

    df = con.execute('SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date').df()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    close = df.set_index('trade_date')['close']

    result_v6 = backtest_v6(signal_v6, close)
    print(f"  年化: {result_v6['ann_ret']:.1%}")
    print(f"  Sharpe: {result_v6['sharpe']:.2f}")
    print(f"  MaxDD: {result_v6['max_dd']:.1%}")
    print(f"  胜率: {result_v6['win_rate']:.1%}")

    # 2. 获取四象限信号
    print("\n[2] 四象限信号:")
    quadrant_signal = compute_quadrant_signal(con)
    print(f"  信号范围: {quadrant_signal.index[0]} ~ {quadrant_signal.index[-1]}")
    print(f"  做多比例: {quadrant_signal.mean():.1%}")

    # 3. 计算四象限的z-score
    print("\n[3] 四象限z-score因子:")
    quadrant_z = rolling_zscore(quadrant_signal, window=52)
    print(f"  因子范围: {quadrant_z.min():.2f} ~ {quadrant_z.max():.2f}")

    # 4. 测试四象限因子与v6.2信号的相关性
    print("\n[4] 相关性分析:")
    common = signal_v6.index.intersection(quadrant_z.dropna().index)
    v6_pos = signal_v6.loc[common, 'position']
    quad_z = quadrant_z.reindex(common)

    corr = v6_pos.corr(quad_z)
    print(f"  v6.2仓位 vs 四象限z-score: {corr:.3f}")

    # 5. 测试组合策略: v6.2 + 四象限因子加权
    print("\n[5] 组合策略测试:")

    # 对齐
    close_w = close.resample('W-FRI').last()
    common = signal_v6.index.intersection(quadrant_signal.index).intersection(close_w.index)
    v6_pos = signal_v6.loc[common, 'position'].fillna(0)
    quad_sig = quadrant_signal.reindex(common).fillna(0)
    close_w = close_w.reindex(common)

    # 组合1: v6.2 + 四象限信号 (线性组合)
    combo1_pos = (v6_pos * 0.7 + quad_sig * 0.3).clip(0, 1)

    # 组合2: v6.2 * 四象限信号 (乘法组合)
    combo2_pos = (v6_pos * quad_sig).clip(0, 1)

    # 组合3: v6.2 仅在四象限做多时持仓
    combo3_pos = v6_pos.copy()
    combo3_pos[quad_sig < 0.5] = 0

    ret = close_w.pct_change()

    for name, pos in [("v6.2+四象限 70/30", combo1_pos),
                       ("v6.2*四象限", combo2_pos),
                       ("v6.2 (四象限过滤)", combo3_pos)]:
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
    test_v5_with_quadrant_factor()
