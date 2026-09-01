# coding=utf-8
"""weekly_backtest.py — 通用周频回测引擎.

数据源无关: 接收价格 + 仓位序列作为参数.

核心逻辑:
    - 计算周频收益: pos.shift(1) * ret - 交易成本
    - 输出: 年化收益/年化波动/Sharpe/MaxDD/胜率/活跃胜率

参考实现:
    - QuantNodes/strategy/commodity_fundamental/gold/gold_utils.py (backtest_weekly)
    - QuantNodes/strategy/commodity_fundamental/copper/copper_momentum.py (backtest_momentum)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_weekly(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 5.0,
    dd_stop: float | None = None,
) -> dict:
    """通用周频回测.

    Args:
        close: 周频收盘价
        position: 周频仓位 [0, 1]
        cost_bps: 交易成本 (基点)
        dd_stop: 回撤止损阈值 (如 -0.15, None则不启用)

    Returns:
        dict with keys:
            - n_weeks: 回测周数
            - ann_ret: 年化收益
            - ann_vol: 年化波动
            - sharpe: Sharpe比率
            - max_dd: 最大回撤
            - win_rate: 胜率 (含空仓周)
            - win_rate_active: 胜率 (仅开仓周)
            - avg_pos: 平均仓位
            - pct_active: 开仓时间占比
            - nav: 净值曲线
    """
    if len(close) == 0:
        return {
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "win_rate_active": 0, "avg_pos": 0, "pct_active": 0,
            "nav": pd.Series(dtype=float),
        }

    ret = close.pct_change()
    strat_ret = position.shift(1) * ret
    cost = position.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "win_rate_active": 0, "avg_pos": 0, "pct_active": 0,
            "nav": pd.Series(dtype=float),
        }

    if dd_stop is not None:
        for i in range(1, len(nav)):
            current_dd = nav.iloc[i] / nav.iloc[:i + 1].max() - 1
            if current_dd < dd_stop:
                strat_ret.iloc[i] = 0
        nav = (1 + strat_ret).cumprod()

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    pos_aligned = position.shift(1).reindex(strat_ret.index).fillna(0)
    active = strat_ret[pos_aligned > 0]

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "win_rate_active": (active > 0).mean() if len(active) > 0 else 0,
        "avg_pos": position.mean(),
        "pct_active": (position > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(
    close: pd.Series,
    position: pd.Series,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> pd.Series:
    """获取周频收益序列 (用于组合优化).

    Args:
        close: 周频收盘价
        position: 周频仓位
        start: 起始日期
        end: 截止日期

    Returns:
        周频策略收益
    """
    ret = close.pct_change()
    strat_ret = position.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "backtest_weekly",
    "get_weekly_returns",
]
