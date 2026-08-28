#!/usr/bin/env python3
"""scripts/cmf/test_inventory_cycle_weekly_stop.py

测试四象限策略 + 周频止损
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_optimized,
    backtest_inventory_cycle_optimized,
)

DB = Path("data/cmf/fundamental.duckdb")


def compute_weekly_atr(close_w: pd.Series, high_w: pd.Series, low_w: pd.Series,
                       window: int = 4) -> pd.Series:
    """计算周频ATR."""
    tr = pd.DataFrame({
        'hl': high_w - low_w,
        'hc': abs(high_w - close_w.shift(1)),
        'lc': abs(low_w - close_w.shift(1)),
    }).max(axis=1)
    return tr.rolling(window).mean()


def apply_weekly_atr_stop(
    signal: pd.DataFrame,
    atr_multiple: float = 3.0,
    atr_window: int = 4,
) -> pd.DataFrame:
    """应用周频ATR止损.

    参数:
        signal: compute_inventory_cycle_optimized输出
        atr_multiple: ATR倍数 (默认3.0)
        atr_window: ATR窗口 (默认4周)
    """
    close_w = signal["weekly_ret"]
    pos = signal["position"]

    # 计算周频ATR (使用收益率的波动率)
    ret_vol = close_w.rolling(atr_window).std() * np.sqrt(52)
    atr = ret_vol  # 使用收益率波动率作为ATR

    # 应用止损
    new_pos = pos.copy()
    current_stop = np.nan

    for i in range(1, len(signal)):
        # 信号变化时重置止损
        if pos.iloc[i] != pos.iloc[i-1]:
            if pos.iloc[i] > 0:
                current_stop = close_w.iloc[i] - atr_multiple * atr.iloc[i]
            elif pos.iloc[i] < 0:
                current_stop = close_w.iloc[i] + atr_multiple * atr.iloc[i]
            else:
                current_stop = np.nan

        # 更新止损线
        if pos.iloc[i] > 0 and not np.isnan(atr.iloc[i]):
            new_stop = close_w.iloc[i] - atr_multiple * atr.iloc[i]
            current_stop = max(current_stop, new_stop)
        elif pos.iloc[i] < 0 and not np.isnan(atr.iloc[i]):
            new_stop = close_w.iloc[i] + atr_multiple * atr.iloc[i]
            current_stop = min(current_stop, new_stop)

        # 检查触发
        if pos.iloc[i] > 0 and not np.isnan(current_stop):
            if close_w.iloc[i] <= current_stop:
                new_pos.iloc[i] = 0.0
                current_stop = np.nan
        elif pos.iloc[i] < 0 and not np.isnan(current_stop):
            if close_w.iloc[i] >= current_stop:
                new_pos.iloc[i] = 0.0
                current_stop = np.nan

    signal["position_with_stop"] = new_pos
    return signal


def backtest_weekly_stop(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.10,
    target_vol: float = 0.12,
) -> dict:
    """回测周频止损策略."""
    weekly_ret = signal["weekly_ret"]
    pos = signal["position_with_stop"]

    # 波动率目标化
    vol_4w = weekly_ret.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(-1, 1)

    # 回测
    strat_ret = pos.shift(1) * weekly_ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损: 回撤超阈值时清仓
    nav = (1 + strat_ret).cumprod()
    for i in range(1, len(nav)):
        current_dd = nav.iloc[i] / nav.iloc[:i+1].max() - 1
        if current_dd < dd_threshold:
            strat_ret.iloc[i] = 0

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def test_weekly_stop():
    """测试周频止损效果."""
    print("=" * 60)
    print("测试周频止损效果")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 测试不同的ATR倍数
    atr_multiples = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    results = []
    for atr_mult in atr_multiples:
        signal = compute_inventory_cycle_optimized(con=con)
        signal = apply_weekly_atr_stop(signal, atr_multiple=atr_mult)
        result = backtest_weekly_stop(signal)
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


def compare_strategies():
    """对比所有策略."""
    print("\n" + "=" * 60)
    print("对比所有策略")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 1. 无止损 (周频)
    signal_opt = compute_inventory_cycle_optimized(con=con)
    result_opt = backtest_inventory_cycle_optimized(signal_opt)

    # 2. 周频止损 (最优)
    signal_weekly = compute_inventory_cycle_optimized(con=con)
    signal_weekly = apply_weekly_atr_stop(signal_weekly, atr_multiple=3.0)
    result_weekly = backtest_weekly_stop(signal_weekly)

    # 3. 日频止损 (之前测试)
    from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
        compute_inventory_cycle_with_atr_stop,
        backtest_inventory_cycle_with_atr_stop,
    )
    signal_daily = compute_inventory_cycle_with_atr_stop(con=con, atr_multiple=2.0)
    result_daily = backtest_inventory_cycle_with_atr_stop(signal_daily)

    print(f"\n{'策略':<20} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8} {'频率':<8}")
    print("-" * 60)
    print(f"{'无止损':<20} {result_opt['ann_ret']:<8.1%} {result_opt['sharpe']:<8.2f} "
          f"{result_opt['max_dd']:<8.1%} {result_opt['win_rate']:<8.1%} {'周频':<8}")
    print(f"{'周频止损(3.0)':<20} {result_weekly['ann_ret']:<8.1%} {result_weekly['sharpe']:<8.2f} "
          f"{result_weekly['max_dd']:<8.1%} {result_weekly['win_rate']:<8.1%} {'周频':<8}")
    print(f"{'日频止损(2.0)':<20} {result_daily['ann_ret']:<8.1%} {result_daily['sharpe']:<8.2f} "
          f"{result_daily['max_dd']:<8.1%} {result_daily['win_rate']:<8.1%} {'日频':<8}")

    return result_opt, result_weekly, result_daily


if __name__ == "__main__":
    print("四象限策略 + 周频止损测试")
    print("=" * 60)

    # 测试周频止损
    test_weekly_stop()

    # 对比所有策略
    compare_strategies()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
