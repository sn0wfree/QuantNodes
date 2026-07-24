# coding=utf-8
"""v9 完整回测引擎.

功能:
    1. 计算周组合收益
    2. 扣除交易成本
    3. 累积 NAV
    4. 计算关键指标 (Sharpe, Calmar, MaxDD, AnnRet)
    5. 多起点 + 滚动窗口 + 成本敏感性
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weekly_returns(weights: pd.DataFrame, weekly_returns: pd.DataFrame) -> pd.Series:
    """计算周组合收益.

    参数:
        weights: (T, N) 周权重 (周初建仓)
        weekly_returns: (T, N) 周收益 (周末结算)

    返回:
        port_returns: (T,) 周组合收益
    """
    common = weights.index.intersection(weekly_returns.index)
    w = weights.loc[common]
    r = weekly_returns.loc[common]
    return (w * r).sum(axis=1)


def apply_transaction_cost(
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """计算调仓成本.

    参数:
        weights: (T, N) 权重时序
        cost_bps: 单边交易成本 (bp)

    返回:
        cost_returns: (T,) 成本收益 (负数, 调仓时扣除)
    """
    w_diff = weights.diff().abs().sum(axis=1)
    cost_pct = w_diff * cost_bps / 10000.0
    return -cost_pct.fillna(0)


def compute_nav(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """累积 NAV.

    参数:
        returns: 周收益
        initial: 初始净值

    返回:
        nav: NAV 时序
    """
    nav = initial * (1 + returns).cumprod()
    nav.iloc[0] = initial
    return nav


def compute_metrics(
    returns: pd.Series,
    freq: str = "W",
    rf: float = 0.02,
) -> dict:
    """计算回测指标.

    参数:
        returns: 收益时序
        freq: 'D' | 'W' | 'M'
        rf: 无风险利率 (年化)

    返回:
        dict: {Sharpe, Calmar, MaxDD, AnnRet, Vol, WinRate, ...}
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {}

    freq_map = {"D": 252, "W": 52, "M": 12}
    periods = freq_map.get(freq, 52)

    excess = returns - rf / periods
    if excess.std() == 1e-10 or np.isnan(excess.std()):
        sharpe = 0.0
    else:
        sharpe = float(excess.mean() / excess.std() * np.sqrt(periods))
        if not np.isfinite(sharpe):
            sharpe = 0.0

    nav = compute_nav(returns)
    if nav.empty:
        return {}

    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    max_dd = float(drawdown.min())

    total_return = (nav.iloc[-1] / nav.iloc[0]) - 1 if nav.iloc[0] > 0 else 0
    n_years = len(returns) / periods
    if n_years > 0 and (1 + total_return) > 0:
        ann_ret = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_ret = 0.0

    if abs(max_dd) > 1e-6:
        calmar = ann_ret / abs(max_dd)
    else:
        calmar = 0.0

    vol = returns.std() * np.sqrt(periods)
    win_rate = (returns > 0).mean()

    return {
        "Sharpe": float(sharpe),
        "Calmar": float(calmar),
        "MaxDD": float(max_dd),
        "AnnRet": float(ann_ret),
        "Vol": float(vol),
        "WinRate": float(win_rate),
        "TotalReturn": float(total_return),
        "N_Weeks": int(len(returns)),
        "N_Years": float(n_years),
    }


def run_backtest(
    weights: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    cost_bps: float = 10.0,
    initial_nav: float = 1.0,
    freq: str = "W",
) -> tuple:
    """主回测函数.

    参数:
        weights: (T, N) 权重
        weekly_returns: (T, N) 收益 (日频或周频)
        cost_bps: 单边交易成本 (bp)
        initial_nav: 初始净值
        freq: 收益频率 ('D'=日频, 'W'=周频, 'M'=月频)

    返回:
        (nav, returns, metrics)
    """
    common = weights.index.intersection(weekly_returns.index)
    w = weights.loc[common].fillna(0)
    r = weekly_returns.loc[common].fillna(0)

    port_returns = compute_weekly_returns(w, r)
    cost_returns = apply_transaction_cost(w, cost_bps=cost_bps)

    net_returns = port_returns + cost_returns

    nav = compute_nav(net_returns, initial=initial_nav)
    metrics = compute_metrics(net_returns, freq=freq)

    return nav, net_returns, metrics


def run_backtest_long_short(
    long_weights: pd.DataFrame,
    short_weights: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> tuple:
    """多空回测 (用于多空择时).

    参数:
        long_weights: (T, N) 多头权重
        short_weights: (T, N) 空头权重 (正数表示权重)
        weekly_returns: (T, N) 周收益

    返回:
        (nav, returns, metrics)
    """
    common = long_weights.index.intersection(weekly_returns.index)
    lw = long_weights.loc[common].fillna(0)
    sw = short_weights.loc[common].fillna(0)
    r = weekly_returns.loc[common]

    long_ret = (lw * r).sum(axis=1)
    short_ret = -(sw * r).sum(axis=1)

    gross_ret = long_ret + short_ret

    turnover = (lw.diff().abs().sum(axis=1) + sw.diff().abs().sum(axis=1)).fillna(0)
    cost_ret = -turnover * cost_bps / 10000.0

    net_ret = gross_ret + cost_ret
    nav = compute_nav(net_ret)
    metrics = compute_metrics(net_ret)

    return nav, net_ret, metrics


def multi_start_backtest(
    weights_func,
    weekly_returns: pd.DataFrame,
    start_dates: list,
    end_date: pd.Timestamp,
    min_train_weeks: int = 208,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """多起点回测.

    参数:
        weights_func: callable(start, end) -> DataFrame
        weekly_returns: (T, N) 周收益
        start_dates: 起点列表
        end_date: 统一终点
        min_train_weeks: 最小训练期
        cost_bps: 单边成本

    返回:
        DataFrame: 每行一个起点的指标
    """
    results = []
    for start in start_dates:
        if (end_date - start).days / 7 < min_train_weeks:
            continue

        try:
            weights = weights_func(pd.Timestamp(start), end_date)
            nav, returns, metrics = run_backtest(weights, weekly_returns, cost_bps=cost_bps)
            metrics["start_date"] = str(start.date())
            results.append(metrics)
        except Exception as e:
            print(f"  起点 {start} 失败: {e}")

    return pd.DataFrame(results)


def cost_sensitivity(
    weights: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    cost_bps_list: list[float] | None = None,
) -> pd.DataFrame:
    """成本敏感性测试.

    参数:
        weights: (T, N) 周权重
        weekly_returns: (T, N) 周收益
        cost_bps_list: 成本列表 (默认 [5, 10, 20, 50])

    返回:
        DataFrame: 每行一个成本的指标
    """
    if cost_bps_list is None:
        cost_bps_list = [5, 10, 20, 50]

    results = []
    for cost in cost_bps_list:
        nav, returns, metrics = run_backtest(weights, weekly_returns, cost_bps=cost)
        metrics["cost_bps"] = cost
        results.append(metrics)

    return pd.DataFrame(results)


def compare_strategies(
    strategies: dict,
    weekly_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """多策略对比.

    参数:
        strategies: {name: weights}
        weekly_returns: (T, N) 周收益
        cost_bps: 单边成本

    返回:
        DataFrame: 每行一个策略的指标
    """
    results = []
    for name, weights in strategies.items():
        nav, returns, metrics = run_backtest(weights, weekly_returns, cost_bps=cost_bps)
        metrics["strategy"] = name
        results.append(metrics)

    return pd.DataFrame(results)