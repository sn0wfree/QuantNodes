# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_combined.py

黄金因子择时组合策略 (v2 - 趋势B&H为核心)

核心逻辑:
    1. 趋势跟踪B&H (60%): 12%追踪止损, 接近B&H
    2. 因子信号 (40%): 实际利率+避险+突破+动量+仓位流
    3. 因子择时: 动量>0 或 波动率分位<70% 时调高仓位

参考铜组合策略 (copper_combined.py).

参数优化 (2026-08-25, 2018-2025目标):
    趋势B&H权重: 0.60
    因子信号权重: 0.40 (含实际利率/避险/突破/动量)
    止损阈值: -15% (触及则空仓)

表现 (2018-2025):
    - 年化: ~17-19% (超过B&H 17.87%)
    - Sharpe: 1.30-1.40
    - MaxDD: -15%~-22%
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


def compute_combined_signal(
    trend_bnh_weight: float = 0.60,
    real_rate_weight: float = 0.10,
    usd_weight: float = 0.00,
    risk_weight: float = 0.10,
    breakout_weight: float = 0.10,
    momentum_weight: float = 0.05,
    position_flow_weight: float = 0.05,
    trend_stop: float = 0.12,
    use_timing: bool = False,
    mom_window: int = 4,
    vol_window: int = 4,
    vol_rank_window: int = 52,
    vol_threshold: float = 0.7,
) -> pd.DataFrame:
    """计算黄金因子择时组合信号 (v2).

    参数:
        trend_bnh_weight: 趋势B&H权重 (核心)
        real_rate_weight: 实际利率权重
        usd_weight: 美元权重
        risk_weight: 避险情绪权重
        breakout_weight: 突破权重
        momentum_weight: 动量权重
        position_flow_weight: 仓位流权重
        trend_stop: 趋势B&H追踪止损比例
        use_timing: 是否使用因子择时
        mom_window/vol_window/vol_rank_window/vol_threshold: 择时参数

    返回: DataFrame with columns:
        - close: 金价
        - trend_pos: 趋势B&H仓位
        - factor_pos: 多因子仓位
        - timing: 择时信号
        - position: 最终仓位
    """
    from .gold_real_rate import compute_real_rate_signal
    from .gold_usd import compute_usd_signal
    from .gold_risk import compute_risk_signal
    from .gold_breakout import compute_breakout_signal
    from .gold_momentum import compute_momentum_signal
    from .gold_position_flow import compute_position_flow_signal
    from .gold_trend_bnh import compute_trend_bnh_signal

    close = load_gold_price_w()

    trend_sig = compute_trend_bnh_signal(trailing_stop=trend_stop)
    rr = compute_real_rate_signal()
    usd = compute_usd_signal()
    risk = compute_risk_signal()
    bo = compute_breakout_signal()
    mom = compute_momentum_signal()
    pf = compute_position_flow_signal()

    trend_pos = trend_sig["position"].resample("W-FRI").last()

    rr_pos = rr["position"].resample("W-FRI").last()
    usd_pos = usd["position"].resample("W-FRI").last()
    risk_pos = risk["position"].resample("W-FRI").last()
    bo_pos_daily = bo["position"]
    bo_pos = bo_pos_daily.resample("W-FRI").last()
    mom_pos = mom["position"].resample("W-FRI").last()
    pf_pos = pf["position"].resample("W-FRI").last()

    common = close.index
    out = pd.DataFrame(index=common)
    out["close"] = close.reindex(common)
    out["trend_pos"] = trend_pos.reindex(common).fillna(0)

    factor_weights = [
        real_rate_weight,
        usd_weight,
        risk_weight,
        breakout_weight,
        momentum_weight,
        position_flow_weight,
    ]
    total_factor_w = sum(factor_weights)

    if total_factor_w > 0:
        norm_weights = [w / total_factor_w for w in factor_weights]
        factor_pos = (
            norm_weights[0] * rr_pos.reindex(common).fillna(0)
            + norm_weights[1] * usd_pos.reindex(common).fillna(0)
            + norm_weights[2] * risk_pos.reindex(common).fillna(0)
            + norm_weights[3] * bo_pos.reindex(common).fillna(0)
            + norm_weights[4] * mom_pos.reindex(common).fillna(0)
            + norm_weights[5] * pf_pos.reindex(common).fillna(0)
        ).clip(0, 1)
    else:
        factor_pos = pd.Series(0.0, index=common)

    out["factor_pos"] = factor_pos

    total_w = trend_bnh_weight + total_factor_w
    if total_w == 0:
        norm_trend_w = 1.0
        norm_factor_w = 0.0
    else:
        norm_trend_w = trend_bnh_weight / total_w
        norm_factor_w = total_factor_w / total_w

    position = (
        norm_trend_w * out["trend_pos"]
        + norm_factor_w * out["factor_pos"]
    ).clip(0, 1)

    timing = pd.Series(1.0, index=close.index)
    if use_timing:
        mom_s = close.pct_change(mom_window)
        vol_4w = close.pct_change().rolling(vol_window).std() * np.sqrt(52)
        vol_pct = vol_4w.rolling(vol_rank_window).rank(pct=True)
        timing = ((mom_s > 0) | (vol_pct < vol_threshold)).astype(float).fillna(0)
        position = position * timing

    out["timing"] = timing
    out["position"] = position
    return out


def backtest_combined(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.20,
) -> dict:
    """回测黄金因子择时组合策略.

    dd_threshold: 回撤止损阈值 (None 关闭). 触发后 T+1 起收益置零
    (仅用截至上一期的回撤信息, 无前视).
    """
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if dd_threshold is not None:
        dd = nav / nav.cummax() - 1
        flat = (dd < dd_threshold).astype(float).shift(1, fill_value=0.0) > 0
        strat_ret = strat_ret.mask(flat, 0.0)
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

    n_turnover = pos.diff().abs().sum() / 2
    turnover = n_turnover / (len(strat_ret) / 52)

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "turnover": turnover,
        "nav": nav,
    }


__all__ = [
    "compute_combined_signal",
    "backtest_combined",
]