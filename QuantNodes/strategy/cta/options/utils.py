# coding=utf-8
"""期权工具函数 — 从 期权工作平台/op1.py + loading_quote.py 移植."""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def required_share(require_hedged_size: float, delta: float, f: float) -> float:
    """计算对冲所需手数: hedge_size / delta / f / 100.

    Args:
        require_hedged_size: 目标对冲名义 (RMB)
        delta: 期权 delta
        f: 标的价格
    """
    return require_hedged_size / delta / f / 100


def cal_pnl(f: float, selected_with_weight: pd.DataFrame) -> float:
    """计算期权组合在标的价格 f 时的 P&L.

    Args:
        f: 标的结算价
        selected_with_weight: DataFrame 含 k, cp, weight 列

    Returns:
        总 P&L
    """
    f_k = f - selected_with_weight["k"]

    call_mask = selected_with_weight["cp"] == "C"
    call_pnl = np.dot(
        (f_k[call_mask] >= 0) * f_k[call_mask],
        selected_with_weight[call_mask]["weight"],
    )

    put_mask = selected_with_weight["cp"] == "P"
    put_pnl = np.dot(
        (f_k[put_mask] <= 0) * f_k[put_mask] * -1,
        selected_with_weight[put_mask]["weight"],
    )

    return call_pnl + put_pnl


def cal_portfolio_greek(
    weight, Delta, Gamma, Vega, Theta, Rho
) -> pd.DataFrame:
    """计算组合希腊字母.

    Args:
        weight: 权重数组
        Delta/Gamma/Vega/Theta/Rho: 矩阵 (index=strike, columns=contract)

    Returns:
        DataFrame 含 Delta/Gamma/Vega/Theta/Rho 列
    """
    d = Delta.dot(weight).to_frame("Delta")
    g = Gamma.dot(weight).to_frame("Gamma")
    v = Vega.dot(weight).to_frame("Vega")
    t = Theta.dot(weight).to_frame("Theta")
    r = Rho.dot(weight).to_frame("Rho")
    return pd.concat([d, g, v, t, r], axis=1)


def find_third_friday(year: int, month: int) -> datetime.date:
    """计算指定年月的第三个周五 (CFFEX 期权到期日规则).

    Args:
        year: 年份
        month: 月份

    Returns:
        第三个周五日期
    """
    first_day = datetime.date(year, month, 1)
    first_friday = first_day + datetime.timedelta(days=(4 - first_day.weekday()))
    return first_friday + datetime.timedelta(weeks=2)


def ym_to_expire_day(
    ym: str, working_days: pd.DatetimeIndex, year_prefix: str = "20"
) -> pd.Timestamp:
    """年月代码 → 到期日 (顺延到最近交易日).

    Args:
        ym: 年月代码 (如 '2503')
        working_days: 交易日历
        year_prefix: 年份前缀

    Returns:
        到期日 Timestamp
    """
    year = int(year_prefix + str(ym)[:2])
    m = int(str(ym)[2:])
    day = pd.to_datetime(find_third_friday(year, m))
    if day in working_days:
        return day
    return working_days[working_days >= day].head(1)
