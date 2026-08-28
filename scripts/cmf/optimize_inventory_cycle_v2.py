#!/usr/bin/env python3
"""scripts/cmf/optimize_inventory_cycle_v2.py

优化库存周期四象限策略v2

优化点:
1. 库存数据: 使用EMA代替简单周频采样，保留更多信息
2. 价格数据: 使用收益率序列处理换月跳价
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    load_lme_inventory,
    backtest_inventory_cycle,
    analyze_quadrant_performance,
)

DB = Path("data/cmf/fundamental.duckdb")


def compute_inventory_cycle_v2_optimized(
    con=None,
    econ_window: int = 52,
    inv_change_window: int = 13,
    inv_ema_span: int = 5,
    use_ema: bool = True,
    econ_threshold: float = 0.0,
    inv_threshold: float = 0.0,
    use_short: bool = False,
) -> pd.DataFrame:
    """优化版四象限信号.

    优化点:
        1. 库存数据: EMA平滑，保留日内趋势
        2. 价格数据: 使用收益率，避免换月跳价影响

    参数:
        con: 数据库连接
        econ_window: 经济增长方向判断窗口 (周)
        inv_change_window: 库存变化窗口 (周)
        inv_ema_span: 库存EMA平滑窗口 (天)
        use_ema: 是否使用EMA
        econ_threshold: 经济增长方向阈值
        inv_threshold: 库存变化方向阈值
        use_short: 是否允许做空
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格数据 - 使用收益率序列
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    df_price = df_price.set_index("trade_date")

    # 计算日收益率，再转周频
    daily_ret = df_price["close"].pct_change()
    weekly_ret = daily_ret.resample("W-FRI").sum()  # 周收益率 = 日收益率之和

    # 加载库存数据
    inv = load_lme_inventory(con)

    # 库存数据处理: EMA平滑
    if use_ema:
        inv_smooth = inv.ewm(span=inv_ema_span).mean()
    else:
        inv_smooth = inv

    inv_w = inv_smooth.resample("W-FRI").last()

    # 对齐数据
    common = weekly_ret.index.intersection(inv_w.index)
    weekly_ret = weekly_ret.reindex(common)
    inv_w = inv_w.reindex(common)

    # 计算经济增长方向: 使用累积收益率
    cum_ret = (1 + weekly_ret).cumprod()
    econ_momentum = cum_ret.pct_change(econ_window)
    econ_up = econ_momentum > econ_threshold

    # 计算库存变化方向
    inv_change = inv_w.pct_change(inv_change_window)
    inv_up = inv_change > inv_threshold

    # 四象限分类
    quadrant = pd.Series(0, index=common)
    quadrant[econ_up & ~inv_up] = 1    # 被动去库: 做多 (最强)
    quadrant[econ_up & inv_up] = 2     # 主动补库: 做多
    quadrant[~econ_up & inv_up] = 3    # 被动补库: 做空
    quadrant[~econ_up & ~inv_up] = 4   # 主动去库: 观望

    # 仓位信号
    position = pd.Series(0.0, index=common)
    position[quadrant == 1] = 1.0      # 被动去库: 做多
    position[quadrant == 2] = 1.0      # 主动补库: 做多
    if use_short:
        position[quadrant == 3] = -1.0  # 被动补库: 做空
    else:
        position[quadrant == 3] = 0.0   # 被动补库: 空仓
    position[quadrant == 4] = 0.0       # 主动去库: 观望

    return pd.DataFrame({
        "weekly_ret": weekly_ret,
        "inventory": inv_w,
        "econ_momentum": econ_momentum,
        "inv_change": inv_change,
        "quadrant": quadrant,
        "position": position,
    })


def backtest_inventory_cycle_v2(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.10,
    target_vol: float = 0.12,
) -> dict:
    """回测优化版四象限策略.

    参数:
        signal: compute_inventory_cycle_v2_optimized输出
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
        target_vol: 目标波动率
    """
    weekly_ret = signal["weekly_ret"]
    pos = signal["position"]

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


def test_ema_optimization():
    """测试EMA优化效果."""
    print("=" * 60)
    print("测试EMA优化效果")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 测试不同的EMA窗口
    ema_spans = [1, 3, 5, 7, 10, 14]

    results = []
    for span in ema_spans:
        signal = compute_inventory_cycle_v2_optimized(
            con=con,
            inv_ema_span=span,
            use_ema=True,
        )
        result = backtest_inventory_cycle_v2(signal)
        result["ema_span"] = span
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'EMA窗口':<10} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 40)

    for _, row in df.iterrows():
        print(f"{row['ema_span']:<10} {row['ann_ret']:<8.1%} {row['sharpe']:<8.2f} "
              f"{row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    # 最优EMA窗口
    best = df.iloc[0]
    print(f"\n最优EMA窗口: {best['ema_span']}")
    print(f"  年化: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MaxDD: {best['max_dd']:.1%}")

    return df


def compare_with_original():
    """与原始策略对比."""
    print("\n" + "=" * 60)
    print("与原始策略对比")
    print("=" * 60)

    from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
        compute_inventory_cycle_quadrant,
        backtest_inventory_cycle,
    )

    con = duckdb.connect(str(DB), read_only=True)

    # 原始策略
    signal_orig = compute_inventory_cycle_quadrant()
    result_orig = backtest_inventory_cycle(signal_orig)

    # 优化策略 (最优EMA窗口)
    signal_opt = compute_inventory_cycle_v2_optimized(
        con=con,
        inv_ema_span=5,
        use_ema=True,
    )
    result_opt = backtest_inventory_cycle_v2(signal_opt)

    print(f"\n{'策略':<20} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 50)
    print(f"{'原始策略':<20} {result_orig['ann_ret']:<8.1%} {result_orig['sharpe']:<8.2f} "
          f"{result_orig['max_dd']:<8.1%} {result_orig['win_rate']:<8.1%}")
    print(f"{'优化策略(EMA)':<20} {result_opt['ann_ret']:<8.1%} {result_opt['sharpe']:<8.2f} "
          f"{result_opt['max_dd']:<8.1%} {result_opt['win_rate']:<8.1%}")

    return result_orig, result_opt


def test_overfitting():
    """过拟合检测."""
    print("\n" + "=" * 60)
    print("过拟合检测")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 全样本
    signal_full = compute_inventory_cycle_v2_optimized(con=con, inv_ema_span=5)
    result_full = backtest_inventory_cycle_v2(signal_full)

    # IS (2014-2020)
    signal_is = signal_full[signal_full.index < pd.Timestamp("2021-01-01")]
    result_is = backtest_inventory_cycle_v2(signal_is)

    # OOS (2021-2026)
    signal_oos = signal_full[signal_full.index >= pd.Timestamp("2021-01-01")]
    result_oos = backtest_inventory_cycle_v2(signal_oos)

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
    print("库存周期四象限策略v2优化")
    print("=" * 60)

    # 测试EMA优化
    test_ema_optimization()

    # 与原始策略对比
    compare_with_original()

    # 过拟合检测
    test_overfitting()

    print("\n" + "=" * 60)
    print("优化测试完成!")
    print("=" * 60)
