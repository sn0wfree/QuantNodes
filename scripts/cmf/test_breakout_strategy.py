#!/usr/bin/env python3
"""scripts/cmf/test_breakout_strategy.py

突破策略 + 追踪止损 测试

测试内容:
1. 基准回测
2. 过拟合检测 (IS/OOS)
3. 参数优化
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OPTIONS_DB = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()


def load_copper_price():
    """加载铜价."""
    con = duckdb.connect(str(OPTIONS_DB), read_only=True)
    cu = con.execute("""
        SELECT trade_date, close FROM continuous_main_cu_daily ORDER BY trade_date
    """).df()
    cu["trade_date"] = pd.to_datetime(cu["trade_date"])
    return cu.set_index("trade_date")["close"]


def backtest_breakout(
    cu: pd.Series,
    lookback: int = 20,
    threshold: float = 0.95,
    trailing_stop: float = 0.03,
    cost_bps: float = 5.0,
) -> dict:
    """突破策略回测.

    参数:
        cu: 铜价序列
        lookback: 突破窗口
        threshold: 突破阈值 (接近高点的比例)
        trailing_stop: 追踪止损比例
        cost_bps: 交易成本
    """
    ret = cu.pct_change()

    # 突破信号
    high = cu.rolling(lookback).max()
    signal = (cu > high * threshold).astype(float)

    # 带止损的回测
    pos = pd.Series(0.0, index=cu.index)
    current_pos = 0.0
    peak_price = 0.0

    for i in range(1, len(cu)):
        # 检查是否需要入场
        if signal.iloc[i] > 0 and current_pos == 0:
            current_pos = 1.0
            peak_price = cu.iloc[i]

        # 更新峰值
        if current_pos > 0:
            peak_price = max(peak_price, cu.iloc[i])

            # 追踪止损
            if trailing_stop is not None:
                if cu.iloc[i] < peak_price * (1 - trailing_stop):
                    current_pos = 0.0

        # 检查是否需要出场 (信号变空)
        if signal.iloc[i] == 0 and current_pos > 0:
            current_pos = 0.0

        pos.iloc[i] = current_pos

    # 计算收益
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_days": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "avg_pos": 0, "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "avg_pos": pos.mean(),
        "nav": nav,
    }


def test_overfitting():
    """过拟合检测."""
    print("=" * 60)
    print("突破策略过拟合检测")
    print("=" * 60)

    cu = load_copper_price()

    # 数据分割
    split_date = "2022-01-01"
    cu_is = cu[cu.index < split_date]
    cu_oos = cu[cu.index >= split_date]

    print(f"\n数据范围:")
    print(f"  IS: {cu_is.index[0].date()} ~ {cu_is.index[-1].date()} ({len(cu_is)}天)")
    print(f"  OOS: {cu_oos.index[0].date()} ~ {cu_oos.index[-1].date()} ({len(cu_oos)}天)")

    # IS测试
    print(f"\n[1] IS (2018-2021):")
    result_is = backtest_breakout(cu_is)
    print(f"  年化: {result_is['ann_ret']:.1%}")
    print(f"  Sharpe: {result_is['sharpe']:.2f}")
    print(f"  MaxDD: {result_is['max_dd']:.1%}")
    print(f"  仓位: {result_is['avg_pos']:.0%}")

    # OOS测试
    print(f"\n[2] OOS (2022-2025):")
    result_oos = backtest_breakout(cu_oos)
    print(f"  年化: {result_oos['ann_ret']:.1%}")
    print(f"  Sharpe: {result_oos['sharpe']:.2f}")
    print(f"  MaxDD: {result_oos['max_dd']:.1%}")
    print(f"  仓位: {result_oos['avg_pos']:.0%}")

    # 过拟合判断
    print(f"\n[3] 过拟合判断:")
    sharpe_ratio = result_oos["sharpe"] / result_is["sharpe"] if result_is["sharpe"] != 0 else 0
    print(f"  Sharpe OOS/IS = {sharpe_ratio:.2f}")

    if sharpe_ratio > 0.8:
        print(f"  ✓ 无过拟合 (OOS/IS > 0.8)")
    elif sharpe_ratio > 0.5:
        print(f"  ⚠ 轻度过拟合 (0.5 < OOS/IS < 0.8)")
    else:
        print(f"  ✗ 严重过拟合 (OOS/IS < 0.5)")

    return result_is, result_oos


def optimize_parameters():
    """参数优化."""
    print("\n" + "=" * 60)
    print("突破策略参数优化")
    print("=" * 60)

    cu = load_copper_price()

    # 参数网格
    lookbacks = [10, 15, 20, 25, 30]
    thresholds = [0.90, 0.93, 0.95, 0.97, 0.98]
    trailing_stops = [0.02, 0.03, 0.04, 0.05]

    results = []
    total = len(lookbacks) * len(thresholds) * len(trailing_stops)
    count = 0

    for lb in lookbacks:
        for th in thresholds:
            for ts in trailing_stops:
                count += 1
                result = backtest_breakout(cu, lookback=lb, threshold=th, trailing_stop=ts)
                result["lookback"] = lb
                result["threshold"] = th
                result["trailing_stop"] = ts
                results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\nTop 10 参数组合 (按Sharpe):")
    print(f"\n{'窗口':<6} {'阈值':<6} {'止损':<6} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'仓位':<8}")
    print("-" * 50)

    for _, row in df.head(10).iterrows():
        print(f"{row['lookback']:<6} {row['threshold']:<6.2f} {row['trailing_stop']:<6.2f} "
              f"{row['ann_ret']:<8.1%} {row['sharpe']:<8.2f} {row['max_dd']:<8.1%} {row['avg_pos']:<8.0%}")

    # 最优参数
    best = df.iloc[0]
    print(f"\n最优参数:")
    print(f"  窗口: {best['lookback']}天")
    print(f"  阈值: {best['threshold']:.2f}")
    print(f"  追踪止损: {best['trailing_stop']:.2f}")
    print(f"  年化: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MaxDD: {best['max_dd']:.1%}")

    return df


def main():
    """主函数."""
    # 过拟合检测
    test_overfitting()

    # 参数优化
    optimize_parameters()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
