# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_multi_factor.py

黄金多因子等权组合策略 (扩展版)

核心逻辑:
    6个单因子信号加权, 含3个进攻型因子 + 3个防御型因子

因子:
    进攻型 (capture bull market):
        - 突破 (breakout): 20日高点+3%追踪止损
        - 动量 (momentum): 52周动量>0
        - 央行购金 (cb_flow): 央行储备持续增持
    防御型 (defensive):
        - 实际利率 (real_rate): TIPS MA信号
        - 避险情绪 (risk): VIX+GPRD联合
        - 美元趋势 (usd): DXY动量信号
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last().dropna()


def _resample_daily_pos_to_weekly(pos_daily: pd.Series) -> pd.Series:
    """日度仓位重采样到周度."""
    return pos_daily.resample("W-FRI").last().dropna()


def compute_multi_factor_signal(
    real_rate_weight: float = 0.20,
    usd_weight: float = 0.00,
    risk_weight: float = 0.20,
    breakout_weight: float = 0.25,
    momentum_weight: float = 0.20,
    position_flow_weight: float = 0.15,
    trend_filter_window: int = 0,
) -> pd.DataFrame:
    """计算多因子等权组合信号 (扩展版).

    参数:
        real_rate_weight: 实际利率权重
        usd_weight: 美元权重
        risk_weight: 避险情绪权重
        breakout_weight: 突破权重
        momentum_weight: 动量权重
        position_flow_weight: 仓位流权重
        trend_filter_window: 趋势过滤窗口 (0=不过滤, N=价格>N周MA才持仓)

    返回: DataFrame with columns:
        - close: 金价
        - 各因子仓位
        - position: 综合仓位
    """
    from .gold_real_rate import compute_real_rate_signal
    from .gold_usd import compute_usd_signal
    from .gold_risk import compute_risk_signal
    from .gold_breakout import compute_breakout_signal
    from .gold_momentum import compute_momentum_signal
    from .gold_position_flow import compute_position_flow_signal

    close = load_gold_price_w()

    rr = compute_real_rate_signal()
    usd = compute_usd_signal()
    risk = compute_risk_signal()
    bo = compute_breakout_signal()
    mom = compute_momentum_signal()
    pf = compute_position_flow_signal()

    rr_pos = rr["position"].resample("W-FRI").last()
    usd_pos = usd["position"].resample("W-FRI").last()
    risk_pos = risk["position"].resample("W-FRI").last()
    bo_pos = _resample_daily_pos_to_weekly(bo["position"])
    mom_pos = mom["position"].resample("W-FRI").last()
    pf_pos = pf["position"].resample("W-FRI").last()

    common = close.index
    out = pd.DataFrame(index=common)
    out["close"] = close.reindex(common)
    out["real_rate_pos"] = rr_pos.reindex(common).fillna(0)
    out["usd_pos"] = usd_pos.reindex(common).fillna(0)
    out["risk_pos"] = risk_pos.reindex(common).fillna(0)
    out["breakout_pos"] = bo_pos.reindex(common).fillna(0)
    out["momentum_pos"] = mom_pos.reindex(common).fillna(0)
    out["position_flow_pos"] = pf_pos.reindex(common).fillna(0)

    weights = [
        real_rate_weight,
        usd_weight,
        risk_weight,
        breakout_weight,
        momentum_weight,
        position_flow_weight,
    ]
    total_w = sum(weights)
    if total_w == 0:
        weights = [1 / 6] * 6
    else:
        weights = [w / total_w for w in weights]

    position = (
        weights[0] * out["real_rate_pos"]
        + weights[1] * out["usd_pos"]
        + weights[2] * out["risk_pos"]
        + weights[3] * out["breakout_pos"]
        + weights[4] * out["momentum_pos"]
        + weights[5] * out["position_flow_pos"]
    ).clip(0, 1)

    if trend_filter_window > 0:
        ma = close.rolling(trend_filter_window).mean()
        trend_ok = (close > ma).astype(float).shift(1)
        position = position * trend_ok.reindex(position.index).fillna(0)
        out["trend_filter"] = trend_ok
    else:
        out["trend_filter"] = 1.0

    out["position"] = position
    return out


def backtest_multi_factor(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测多因子组合策略."""
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "avg_pos": 0, "pct_active": 0, "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()
    active = strat_ret[pos.shift(1).reindex(strat_ret.index).fillna(0) > 0]

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "win_rate_active": (active > 0).mean() if len(active) > 0 else 0,
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    **kwargs,
):
    """获取周频收益用于组合优化."""
    signal = compute_multi_factor_signal(**kwargs)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "compute_multi_factor_signal",
    "backtest_multi_factor",
    "get_weekly_returns",
]