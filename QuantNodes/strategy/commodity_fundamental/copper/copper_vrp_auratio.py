# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_vrp_auratio.py

铜VRP和AU-CU价格比策略

策略1: VRP (Volatility Risk Premium)
    - VRP = ATM_IV - HV_20
    - VRP百分位 < 0.30 (期权便宜) 时触发
    - 铜: 20日动量 < 0 → 买入call spread (超卖反弹)

策略2: AU-CU价格比
    - ratio = AU价格 / CU价格 (金铜比)
    - z = (ratio - 均值) / 标准差
    - z > 1.0 → 做多CU (风险偏好回归)
    - z < -1.0 → 空仓 (风险厌恶)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OPTIONS_DB = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
CMF_DB = Path("data/cmf/fundamental.duckdb")


def load_vrp_data(product="CU"):
    """加载VRP数据."""
    con = duckdb.connect(str(OPTIONS_DB), read_only=True)
    df = con.execute(f"""
        SELECT trade_date, vrp, vrp_percentile, atm_iv, hv_20, product_class
        FROM vrp_daily
        WHERE product = '{product}'
        ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")


def load_au_cu_prices():
    """加载AU和CU价格数据."""
    con = duckdb.connect(str(OPTIONS_DB), read_only=True)

    au = con.execute("""
        SELECT trade_date, close FROM continuous_main_au_daily ORDER BY trade_date
    """).df()
    au["trade_date"] = pd.to_datetime(au["trade_date"])
    au = au.set_index("trade_date")["close"]

    cu = con.execute("""
        SELECT trade_date, close FROM continuous_main_cu_daily ORDER BY trade_date
    """).df()
    cu["trade_date"] = pd.to_datetime(cu["trade_date"])
    cu = cu.set_index("trade_date")["close"]

    return au, cu


def compute_vrp_signal(
    vrp_threshold: float = 0.30,
    mom_window: int = 20,
    hold_days: int = 7,
) -> pd.DataFrame:
    """计算VRP信号.

    参数:
        vrp_threshold: VRP百分位阈值 (低于此值触发)
        mom_window: 动量窗口 (天)
        hold_days: 持仓天数

    逻辑:
        - VRP百分位 < vrp_threshold 且 20日动量 < 0 → 做多 (买入call spread)
        - 持仓hold_days天后平仓
    """
    # 加载VRP数据
    vrp = load_vrp_data("CU")

    # 加载CU价格
    con = duckdb.connect(str(OPTIONS_DB), read_only=True)
    cu = con.execute("""
        SELECT trade_date, close FROM continuous_main_cu_daily ORDER BY trade_date
    """).df()
    cu["trade_date"] = pd.to_datetime(cu["trade_date"])
    cu = cu.set_index("trade_date")["close"]

    # 计算动量
    mom = cu.pct_change(mom_window)

    # 对齐
    common = vrp.index.intersection(cu.index).intersection(mom.dropna().index)
    vrp = vrp.reindex(common)
    cu = cu.reindex(common)
    mom = mom.reindex(common)

    # 生成信号
    position = pd.Series(0.0, index=common)
    trigger_count = 0
    hold_counter = 0

    for i in range(len(common)):
        if hold_counter > 0:
            # 持仓中
            position.iloc[i] = 1.0
            hold_counter -= 1
        elif vrp["vrp_percentile"].iloc[i] < vrp_threshold and mom.iloc[i] < 0:
            # 触发条件: VRP便宜 + 超卖
            position.iloc[i] = 1.0
            hold_counter = hold_days - 1
            trigger_count += 1

    return pd.DataFrame({
        "close": cu,
        "vrp": vrp["vrp"],
        "vrp_percentile": vrp["vrp_percentile"],
        "momentum": mom,
        "position": position,
    })


def compute_au_cu_ratio_signal(
    z_window: int = 60,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """计算AU-CU价格比信号.

    参数:
        z_window: z-score窗口 (天)
        z_threshold: z-score阈值

    逻辑:
        - ratio = AU / CU
        - z = (ratio - 均值) / 标准差
        - z > z_threshold → 做多CU (金铜比高, 风险偏好回归)
        - z < -z_threshold → 空仓 (金铜比低, 风险厌恶)
    """
    # 加载价格
    au, cu = load_au_cu_prices()

    # 对齐
    common = au.index.intersection(cu.index)
    au = au.reindex(common)
    cu = cu.reindex(common)

    # 计算金铜比
    ratio = au / cu

    # 计算z-score
    ratio_mean = ratio.rolling(z_window).mean()
    ratio_std = ratio.rolling(z_window).std()
    z = (ratio - ratio_mean) / ratio_std

    # 生成信号
    position = pd.Series(0.0, index=common)
    position[z > z_threshold] = 1.0   # 金铜比高 → 做多CU (风险偏好回归)
    position[z < -z_threshold] = 0.0  # 金铜比低 → 空仓 (风险厌恶)
    # 中间区域: 保持之前的仓位

    # 前向填充
    position = position.replace(0, np.nan).ffill().fillna(0)

    return pd.DataFrame({
        "au_close": au,
        "cu_close": cu,
        "ratio": ratio,
        "z_score": z,
        "position": position,
    })


def backtest_strategy(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.15,
    price_col: str = "close",
) -> dict:
    """回测策略.

    参数:
        signal: 含position和价格列的DataFrame
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
        price_col: 价格列名
    """
    close = signal[price_col]
    pos = signal["position"]
    ret = close.pct_change()

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
            "n_days": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(start="2018-01-01", end="2025-12-31"):
    """获取周频收益用于组合优化."""
    # VRP信号
    vrp_signal = compute_vrp_signal()
    vrp_ret = vrp_signal["close"].pct_change()
    vrp_pos = vrp_signal["position"]
    vrp_strat = vrp_pos.shift(1) * vrp_ret
    vrp_weekly = vrp_strat.resample("W-FRI").sum()

    # AU-CU比信号
    ratio_signal = compute_au_cu_ratio_signal()
    ratio_ret = ratio_signal["cu_close"].pct_change()
    ratio_pos = ratio_signal["position"]
    ratio_strat = ratio_pos.shift(1) * ratio_ret
    ratio_weekly = ratio_strat.resample("W-FRI").sum()

    # 对齐
    common = vrp_weekly.index.intersection(ratio_weekly.index)
    vrp_weekly = vrp_weekly.reindex(common)
    ratio_weekly = ratio_weekly.reindex(common)

    # 截取时间范围
    returns = pd.DataFrame({
        "VRP": vrp_weekly,
        "AU-CU比": ratio_weekly,
    }).dropna()

    returns = returns[(returns.index >= start) & (returns.index <= end)]

    return returns


__all__ = [
    "compute_vrp_signal",
    "compute_au_cu_ratio_signal",
    "backtest_strategy",
    "get_weekly_returns",
]
