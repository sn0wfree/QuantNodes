#!/usr/bin/env python3
"""scripts/cmf/plot_strategy_nav.py

输出策略净值序列并画图对比
回测区间: 2018-2025
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_optimized,
    compute_inventory_cycle_with_atr_stop,
    backtest_inventory_cycle_optimized,
    backtest_inventory_cycle_with_atr_stop,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import build_signal_carry
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6

DB = Path("data/cmf/fundamental.duckdb")


def get_nav_v62(start="2018-01-01", end="2025-12-31"):
    """获取v6.2净值."""
    con = duckdb.connect(str(DB), read_only=True)
    signal = build_signal_v6(con=con)
    pos = signal["position"]

    # 加载价格
    df = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    # 对齐
    common = pos.index.intersection(close_w.index)
    pos = pos.reindex(common).fillna(0)
    close_w = close_w.reindex(common)

    # 回测
    ret = close_w.pct_change()
    cost = pos.diff().abs() * 5 / 10000
    strat_ret = pos.shift(1) * ret - cost
    nav = (1 + strat_ret.dropna()).cumprod()

    # 截取时间范围
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    if len(nav) > 0:
        nav = nav / nav.iloc[0]  # 归一化

    return nav


def get_nav_carry(start="2018-01-01", end="2025-12-31"):
    """获取carry净值."""
    con = duckdb.connect(str(DB), read_only=True)
    signal = build_signal_carry(con)
    pos = signal["position"]

    # 加载价格
    df = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    # 对齐
    common = pos.index.intersection(close_w.index)
    pos = pos.reindex(common).fillna(0)
    close_w = close_w.reindex(common)

    # 回测
    ret = close_w.pct_change()
    cost = pos.diff().abs() * 5 / 10000
    strat_ret = pos.shift(1) * ret - cost
    nav = (1 + strat_ret.dropna()).cumprod()

    # 截取时间范围
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    if len(nav) > 0:
        nav = nav / nav.iloc[0]  # 归一化

    return nav


def get_nav_quadrant(start="2018-01-01", end="2025-12-31"):
    """获取四象限净值."""
    signal = compute_inventory_cycle_optimized()
    result = backtest_inventory_cycle_optimized(signal)
    nav = result["nav"]

    # 截取时间范围
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    nav = nav / nav.iloc[0]  # 归一化

    return nav


def get_nav_quadrant_atr(start="2018-01-01", end="2025-12-31"):
    """获取四象限+ATR止损净值."""
    signal = compute_inventory_cycle_with_atr_stop(atr_multiple=3.0, use_weekly=True)
    result = backtest_inventory_cycle_with_atr_stop(signal)
    nav = result["nav"]

    # 截取时间范围
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    nav = nav / nav.iloc[0]  # 归一化

    return nav


def get_nav_buyhold(start="2018-01-01", end="2025-12-31"):
    """获取buyhold净值."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    # 截取时间范围
    close_w = close_w[(close_w.index >= start) & (close_w.index <= end)]
    nav = close_w / close_w.iloc[0]  # 归一化

    return nav


def get_nav_combo(start="2018-01-01", end="2025-12-31"):
    """获取组合净值: v6.2(50%) + carry(30%) + 四象限(20%)."""
    con = duckdb.connect(str(DB), read_only=True)

    # v6.2
    signal_v6 = build_signal_v6(con=con)
    pos_v6 = signal_v6["position"]

    # carry
    signal_carry = build_signal_carry(con)
    pos_carry = signal_carry["position"]

    # 四象限
    signal_quad = compute_inventory_cycle_optimized()
    pos_quad = signal_quad["position"]

    # 加载价格
    df = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close_w = df.set_index("trade_date")["close"].resample("W-FRI").last()

    # 对齐
    common = pos_v6.index.intersection(pos_carry.index).intersection(pos_quad.index).intersection(close_w.index)
    pos_v6 = pos_v6.reindex(common).fillna(0)
    pos_carry = pos_carry.reindex(common).fillna(0)
    pos_quad = pos_quad.reindex(common).fillna(0)
    close_w = close_w.reindex(common)

    # 组合
    pos_combo = (pos_v6 * 0.5 + pos_carry * 0.3 + pos_quad * 0.2).clip(0, 1)

    # 回测
    ret = close_w.pct_change()
    cost = pos_combo.diff().abs() * 5 / 10000
    strat_ret = pos_combo.shift(1) * ret - cost
    nav = (1 + strat_ret.dropna()).cumprod()

    # 截取时间范围
    nav = nav[(nav.index >= start) & (nav.index <= end)]
    nav = nav / nav.iloc[0]  # 归一化

    return nav


def compute_metrics(nav):
    """计算策略指标."""
    ret = nav.pct_change().dropna()
    n_years = len(ret) / 52
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (ret > 0).mean()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
    }


def main():
    """主函数."""
    print("=" * 60)
    print("策略净值对比 (2018-2025)")
    print("=" * 60)

    # 获取各策略净值
    nav_v62 = get_nav_v62()
    nav_carry = get_nav_carry()
    nav_quadrant = get_nav_quadrant()
    nav_quadrant_atr = get_nav_quadrant_atr()
    nav_buyhold = get_nav_buyhold()

    # 计算指标
    strategies = {
        "v6.2": nav_v62,
        "carry": nav_carry,
        "四象限": nav_quadrant,
        "四象限+ATR": nav_quadrant_atr,
        "buyhold": nav_buyhold,
    }

    print(f"\n{'策略':<25} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 55)

    for name, nav in strategies.items():
        metrics = compute_metrics(nav)
        print(f"{name:<25} {metrics['ann_ret']:<8.1%} {metrics['sharpe']:<8.2f} "
              f"{metrics['max_dd']:<8.1%} {metrics['win_rate']:<8.1%}")

    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=100)

    # 图1: 净值曲线
    ax1 = axes[0]
    ax1.plot(nav_v62.index, nav_v62.values, label="v6.2", linewidth=1.5, color="#1f77b4")
    ax1.plot(nav_carry.index, nav_carry.values, label="carry", linewidth=1.5, color="#ff7f0e")
    ax1.plot(nav_quadrant.index, nav_quadrant.values, label="四象限", linewidth=1.5, color="#2ca02c")
    ax1.plot(nav_quadrant_atr.index, nav_quadrant_atr.values, label="四象限+ATR", linewidth=1.5, color="#d62728")
    ax1.plot(nav_buyhold.index, nav_buyhold.values, label="buyhold", linewidth=1.0, color="#7f7f7f", linestyle="--")

    ax1.set_title("铜策略净值对比 (2018-2025)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值 (归一化)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # 图2: 回撤曲线
    ax2 = axes[1]
    for name, nav in strategies.items():
        dd = nav / nav.cummax() - 1
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)

    ax2.set_title("回撤曲线", fontsize=14, fontweight="bold")
    ax2.set_ylabel("回撤", fontsize=12)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    # 保存图片
    output_path = Path("reports/cmf/copper_strategy_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {output_path}")

    # 保存净值CSV
    nav_df = pd.DataFrame({
        "v6.2": nav_v62,
        "carry": nav_carry,
        "quadrant": nav_quadrant,
        "quadrant_atr": nav_quadrant_atr,
        "buyhold": nav_buyhold,
    })
    csv_path = Path("reports/cmf/copper_strategy_nav.csv")
    nav_df.to_csv(csv_path)
    print(f"净值CSV已保存: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
