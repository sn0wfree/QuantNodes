# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_all_factor.py

黄金全因子分层合成策略

核心逻辑:
    5层因子合成 → 评分 → 调节趋势B&H仓位
    - L1 利率动态 (25%): Δtips, Δus10y → 利率边际变化
    - L2 风险情绪 (20%): vix/gprd/epu百分位 → 避险需求
    - L3 实体需求 (20%): comex/cftc/etf/cb → 资金流
    - L4 美元     (10%): dxy z-score → 美元方向
    - L5 通胀状态 (25%): fed_rate/us10y水平 → 宏观regime

三种模式:
    aggressive: 趋势×(0.85+0.15×score) → ~17%/1.37/-17%
    balanced:   趋势×(0.70+0.30×score) → ~15.7%/1.37/-16%
    conservative: 趋势×(0.50+0.50×score) → ~14%/1.38/-14%
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def _load_indicator(value_col: str, source: str) -> pd.Series:
    """加载指标到周度 (W-FRI, ffill)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE source='{source}' AND value_col='{value_col}' ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()


def _load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last().dropna()


def _z(s: pd.Series, window: int = 156) -> pd.Series:
    """滚动z-score标准化."""
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def compute_composite_score() -> pd.DataFrame:
    """计算5层合成评分.

    返回: DataFrame with columns:
        - score: 合成评分
        - score_norm: sigmoid归一化评分 [0,1]
        - L1~L5: 各层分数
    """
    close = _load_gold_price_w()

    # shift(1) 修正前视效应: 所有指标延迟1周使用 (周频信号 + 数据发布延迟)
    tips = _load_indicator("tips", "gold_fred").shift(1)
    us10y = _load_indicator("us_10y", "gold_macro").shift(1)
    vix = _load_indicator("vix", "gold_fred").shift(1)
    gprd = _load_indicator("gprd", "gold_risk").shift(1)
    epu = _load_indicator("epu", "gold_fred").shift(1)
    inv = _load_indicator("comex_inv", "gold_position").shift(1)
    cftc = _load_indicator("cftc_net", "gold_position").shift(1)
    etf = _load_indicator("etf_tonnes", "gold_position").shift(1)
    cb = _load_indicator("cb_reserves", "gold_position").shift(1)
    dxy = _load_indicator("dxy", "gold_macro").shift(1)
    fed = _load_indicator("fed_rate", "gold_macro").shift(1)

    idx = close.index

    L1 = (-_z(tips.diff(13)) - _z(us10y.diff(13))) / 2
    L2 = (vix.rolling(52).rank(pct=True) + gprd.rolling(52).rank(pct=True) + epu.rolling(52).rank(pct=True) - 1.5)
    L3 = (-_z(inv) + _z(cftc.pct_change(13)) + _z(etf.pct_change(4)) + _z(cb.pct_change(26))) / 4
    L4 = -_z(dxy)
    L5 = (_z(fed) + _z(us10y)) / 2

    weights = {"L1_rate": 0.25, "L2_risk": 0.20, "L3_flow": 0.20, "L4_usd": 0.10, "L5_regime": 0.25}

    composite = (
        weights["L1_rate"] * L1.reindex(idx)
        + weights["L2_risk"] * L2.reindex(idx)
        + weights["L3_flow"] * L3.reindex(idx)
        + weights["L4_usd"] * L4.reindex(idx)
        + weights["L5_regime"] * L5.reindex(idx)
    ).fillna(0)

    score_norm = 1 / (1 + np.exp(-composite))

    return pd.DataFrame({
        "score": composite,
        "score_norm": score_norm,
        "L1_rate": L1.reindex(idx),
        "L2_risk": L2.reindex(idx),
        "L3_flow": L3.reindex(idx),
        "L4_usd": L4.reindex(idx),
        "L5_regime": L5.reindex(idx),
    })


def compute_all_factor_signal(
    mode: str = "balanced",
    trend_stop: float = 0.12,
) -> pd.DataFrame:
    """计算全因子合成策略信号.

    参数:
        mode: 运行模式
            - "aggressive": 0.85×趋势 + 0.15×评分
            - "balanced": 0.70×趋势 + 0.30×评分
            - "conservative": 0.50×趋势 + 0.50×评分
        trend_stop: 趋势B&H追踪止损比例

    返回: DataFrame with columns:
        - close: 金价
        - trend_pos: 趋势B&H仓位
        - score_norm: 合成评分
        - position: 最终仓位
    """
    from .gold_trend_bnh import compute_trend_bnh_signal

    close = _load_gold_price_w()
    score_df = compute_composite_score()
    trend = compute_trend_bnh_signal(trailing_stop=trend_stop)
    trend_pos = trend["position"].reindex(close.index).fillna(0)
    score_norm = score_df["score_norm"].reindex(close.index).fillna(0.5)

    mode_params = {
        "aggressive": (0.85, 0.15),
        "balanced": (0.70, 0.30),
        "conservative": (0.50, 0.50),
    }
    if mode not in mode_params:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(mode_params.keys())}")
    a, b = mode_params[mode]

    position = (a * trend_pos + b * score_norm).clip(0, 1)

    out = pd.DataFrame({
        "close": close,
        "trend_pos": trend_pos,
        "score_norm": score_norm,
        "position": position,
    })
    out["L1_rate"] = score_df["L1_rate"].reindex(close.index)
    out["L2_risk"] = score_df["L2_risk"].reindex(close.index)
    out["L3_flow"] = score_df["L3_flow"].reindex(close.index)
    out["L4_usd"] = score_df["L4_usd"].reindex(close.index)
    out["L5_regime"] = score_df["L5_regime"].reindex(close.index)
    return out


def backtest_all_factor(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.20,
) -> dict:
    """回测全因子策略.

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


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    mode="balanced",
    trend_stop=0.12,
):
    """获取周频收益."""
    signal = compute_all_factor_signal(mode=mode, trend_stop=trend_stop)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "compute_composite_score",
    "compute_all_factor_signal",
    "backtest_all_factor",
    "get_weekly_returns",
]