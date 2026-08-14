# coding=utf-8
"""IV 批量计算管线 — 从 期权工作平台/loading_quote.py 移植.

使用 CodersWheel.boost_up 并行计算 IV + Greeks.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from CodersWheel.QuickTool.boost_up import boost_up

from ..common.option_greeks import calculate_all_greeks, calculate_iv


def cal_greek(
    underlying_price: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
    risk_free_rate: float = 0.02,
    option_type: str = "call",
) -> tuple:
    """计算单只期权 Greeks.

    Returns:
        (Delta, Gamma, Vega, Theta, Rho)
    """
    result = calculate_all_greeks(
        s=underlying_price,
        k=strike,
        r=risk_free_rate,
        g=dividend_yield,
        sigma=volatility,
        t=time_to_maturity,
        option_type=option_type,
    )
    return result.delta, result.gamma, result.vega, result.theta, result.rho


def _iv_task(f, k, r, t, fee, cp, g=0):
    """单个 IV 计算任务 (供 boost_up 并行)."""
    cp_sign = 1 if cp in ("C", "call") else -1
    try:
        iv, diff = calculate_iv(
            S=f, K=k, r=r, T=t, option_price=fee, cp_sign=cp_sign, g=g,
        )
        return iv, diff
    except Exception:
        return np.nan, np.nan


def create_bs_cal_tasks(
    sub_data: pd.DataFrame,
    underlying_prices: pd.Series,
    ym2exec: dict,
    strike_col: str = "k",
    code_col: str = "contract_code",
    fee_col: str = "fee",
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """创建 IV 计算任务 DataFrame.

    Args:
        sub_data: 期权数据
        underlying_prices: 标的价格序列 (index=日期)
        ym2exec: {年月: 到期日} 映射
        strike_col: 行权价列名
        code_col: 合约代码列名
        fee_col: 权利金列名
        target_cols: 分组列 (如 ['dt', 'ym', 'cp'])

    Returns:
        任务 DataFrame, 含 dt/contract_code/f/k/t/fee/cp/ym 列
    """
    if target_cols is None:
        target_cols = ["dt", "ym", "cp"]

    rows = []
    for (ym, cp, dt), df in sub_data.sort_values(strike_col).groupby(target_cols):
        ttm = (ym2exec.get(ym) - dt).days / 252
        if ttm <= 0:
            continue
        f_val = underlying_prices.get(dt)
        if f_val is None or np.isnan(f_val):
            continue
        for contract_code, k, fee in df[[code_col, strike_col, fee_col]].values:
            rows.append({
                "dt": dt,
                "contract_code": contract_code,
                "f": float(f_val),
                "k": float(k),
                "t": float(ttm),
                "fee": float(fee),
                "cp": cp,
                "ym": ym,
            })
    return pd.DataFrame(rows)


def cal_bs_iv_booster(
    sub_data: pd.DataFrame,
    underlying_prices: pd.Series,
    ym2exec: dict,
    strike_col: str = "k",
    code_col: str = "contract_code",
    fee_col: str = "fee",
    target_cols: list[str] | None = None,
    g: float = 0.0,
    r: float = 0.02,
) -> pd.DataFrame:
    """批量并行计算 IV + Greeks.

    Args:
        sub_data: 期权数据
        underlying_prices: 标的价格序列
        ym2exec: {年月: 到期日} 映射
        g: 股息率
        r: 无风险利率

    Returns:
        DataFrame 含 dt/contract_code/f/k/t/fee/cp/ym/bs_iv/resid 列
    """
    tasks_df = create_bs_cal_tasks(
        sub_data, underlying_prices, ym2exec,
        strike_col=strike_col, code_col=code_col,
        fee_col=fee_col, target_cols=target_cols,
    )
    if tasks_df.empty:
        return tasks_df

    tasks_df["r"] = r
    tasks = tasks_df[["f", "k", "r", "t", "fee", "cp"]].values

    func = partial(_iv_task, g=g)
    results = list(boost_up(func, tasks, star=True))

    bs_iv, resid = zip(*results) if results else ([], [])
    tasks_df["bs_iv"] = bs_iv
    tasks_df["resid"] = np.round(resid, 8)
    tasks_df["dt"] = pd.to_datetime(tasks_df["dt"])

    return tasks_df


def compute_greeks_batch(
    iv_df: pd.DataFrame,
    dividend_yield: float = 0.0,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """从 IV 结果批量计算 Greeks.

    Args:
        iv_df: cal_bs_iv_booster 的输出
        dividend_yield: 股息率

    Returns:
        原 DataFrame, 新增 Delta/Gamma/Vega/Theta/Rho 列
    """
    df = iv_df.copy()
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

    for _, row in df.iterrows():
        try:
            d, g, v, t, r = cal_greek(
                underlying_price=row["f"],
                strike=row["k"],
                volatility=row["bs_iv"],
                time_to_maturity=row["t"],
                dividend_yield=dividend_yield,
                risk_free_rate=risk_free_rate,
                option_type="call" if row["cp"] == "C" else "put",
            )
            deltas.append(d)
            gammas.append(g)
            vegas.append(v)
            thetas.append(t)
            rhos.append(r)
        except Exception:
            deltas.append(np.nan)
            gammas.append(np.nan)
            vegas.append(np.nan)
            thetas.append(np.nan)
            rhos.append(np.nan)

    df["Delta"] = deltas
    df["Gamma"] = gammas
    df["Vega"] = vegas
    df["Theta"] = thetas
    df["Rho"] = rhos

    return df
