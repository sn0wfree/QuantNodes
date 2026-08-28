# coding=utf-8
"""gold_multi_factor.py — 黄金多因子等权组合策略 (扩展版).

核心逻辑:
    6个单因子信号加权, 含3个进攻型因子 + 3个防御型因子

因子:
    进攻型: 突破 / 动量 / 央行购金
    防御型: 实际利率 / 避险情绪 / 美元趋势
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


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
    """计算多因子等权组合信号.

    参数:
        real_rate_weight: 实际利率权重
        usd_weight: 美元权重
        risk_weight: 避险情绪权重
        breakout_weight: 突破权重
        momentum_weight: 动量权重
        position_flow_weight: 仓位流权重
        trend_filter_window: 趋势过滤窗口 (0=不过滤)

    返回: DataFrame with columns:
        - close, 各因子仓位, position
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
        real_rate_weight, usd_weight, risk_weight,
        breakout_weight, momentum_weight, position_flow_weight,
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


def backtest_multi_factor(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测多因子组合策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(start="2018-01-01", end="2025-12-31", **kwargs):
    """获取周频收益用于组合优化."""
    signal = compute_multi_factor_signal(**kwargs)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "compute_multi_factor_signal",
    "backtest_multi_factor",
    "get_weekly_returns",
]
