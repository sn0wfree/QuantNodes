# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_momentum.py

铜时序动量策略

核心逻辑:
    - 价格动量: 过去N周收益率 > 0 → 做多
    - 波动率调整: 高波动时降低仓位

参数:
    - lookback: 动量回看窗口 (默认26周=半年)
    - vol_window: 波动率窗口 (默认13周)
    - vol_target: 目标波动率 (默认12%)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def compute_momentum_signal(
    con=None,
    lookback: int = 26,
    vol_window: int = 13,
    vol_target: float = 0.12,
    use_vol_target: bool = True,
) -> pd.DataFrame:
    """计算铜动量信号.

    参数:
        con: 数据库连接
        lookback: 动量回看窗口 (周)
        vol_window: 波动率窗口 (周)
        vol_target: 目标波动率
        use_vol_target: 是否使用波动率目标化

    返回: DataFrame with columns:
        - close_w: 周频收盘价
        - momentum: 动量信号 (收益率)
        - volatility: 波动率
        - position: 仓位 [0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 计算动量
    momentum = close_w.pct_change(lookback)

    # 计算波动率
    ret = close_w.pct_change()
    volatility = ret.rolling(vol_window).std() * np.sqrt(52)

    # 生成信号: 动量 > 0 → 做多
    position = pd.Series(0.0, index=close_w.index)
    position[momentum > 0] = 1.0

    # 波动率目标化
    if use_vol_target:
        scale = (vol_target / volatility).fillna(1.0).clip(upper=1.0)
        position = position * scale

    return pd.DataFrame({
        "close_w": close_w,
        "momentum": momentum,
        "volatility": volatility,
        "position": position,
    })


def compute_momentum_v2(
    con=None,
    lookback_short: int = 13,
    lookback_long: int = 52,
    vol_window: int = 13,
    vol_target: float = 0.12,
) -> pd.DataFrame:
    """动量策略v2: 双动量 (短期+长期).

    参数:
        con: 数据库连接
        lookback_short: 短期动量窗口 (周)
        lookback_long: 长期动量窗口 (周)
        vol_window: 波动率窗口 (周)
        vol_target: 目标波动率

    逻辑:
        - 短期动量 > 0 且 长期动量 > 0 → 满仓
        - 仅短期动量 > 0 → 半仓
        - 仅长期动量 > 0 → 半仓
        - 都 < 0 → 空仓
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 计算动量
    mom_short = close_w.pct_change(lookback_short)
    mom_long = close_w.pct_change(lookback_long)

    # 计算波动率
    ret = close_w.pct_change()
    volatility = ret.rolling(vol_window).std() * np.sqrt(52)

    # 生成信号
    position = pd.Series(0.0, index=close_w.index)

    # 双动量确认
    both_positive = (mom_short > 0) & (mom_long > 0)
    only_short = (mom_short > 0) & (mom_long <= 0)
    only_long = (mom_short <= 0) & (mom_long > 0)

    position[both_positive] = 1.0
    position[only_short] = 0.5
    position[only_long] = 0.5

    # 波动率目标化
    scale = (vol_target / volatility).fillna(1.0).clip(upper=1.0)
    position = position * scale

    return pd.DataFrame({
        "close_w": close_w,
        "mom_short": mom_short,
        "mom_long": mom_long,
        "volatility": volatility,
        "position": position,
    })


def backtest_momentum(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.15,
) -> dict:
    """回测动量策略.

    参数:
        signal: compute_momentum_signal输出
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
    """
    close_w = signal["close_w"]
    pos = signal["position"]
    ret = close_w.pct_change()

    # 回测
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损
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


def optimize_momentum_params():
    """优化动量参数."""
    print("=" * 60)
    print("动量参数优化")
    print("=" * 60)

    con = duckdb.connect(str(DB), read_only=True)

    # 参数网格
    lookbacks = [4, 8, 13, 20, 26, 39, 52]

    results = []
    for lb in lookbacks:
        signal = compute_momentum_signal(con=con, lookback=lb)
        result = backtest_momentum(signal)
        result["lookback"] = lb
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'回看窗口':<10} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 40)

    for _, row in df.iterrows():
        print(f"{row['lookback']:<10} {row['ann_ret']:<8.1%} {row['sharpe']:<8.2f} "
              f"{row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    best = df.iloc[0]
    print(f"\n最优回看窗口: {best['lookback']}周")
    print(f"  年化: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MaxDD: {best['max_dd']:.1%}")

    return df


__all__ = [
    "compute_momentum_signal",
    "compute_momentum_v2",
    "backtest_momentum",
    "optimize_momentum_params",
]
