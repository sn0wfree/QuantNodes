# coding=utf-8
"""v11 回测引擎 — 完整回测 + 业绩指标.

基于 v10 backtest_v10.py 框架 + ACT-1/2/3 增强.

功能:
    1. 调仓日: 用 V11Strategy 生成的权重
    2. 非调仓日: 用前一日权重累积
    3. 调仓成本: 5bp 单边
    4. NAV 计算: 含成本
    5. 业绩指标: Sharpe, Calmar, MaxDD, AnnRet, Vol, WinRate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config_v11 import V10Config as V11Config
from .v11_strategy import V11Strategy


@dataclass
class V11BacktestResult:
    """v11 回测结果."""
    nav: pd.Series                          # NAV 时序
    weights: pd.DataFrame                   # 调仓日权重
    macro_score: pd.Series | None = None
    regime_state: pd.Series | None = None
    bear_prob: pd.Series | None = None
    position_size: pd.Series | None = None
    metrics: dict = field(default_factory=dict)


def _safe_return(a: float, b: float, max_ret: float = 0.5) -> float:
    """安全的单期收益率 (NaN-safe, 限制极端值)."""
    if pd.isna(a) or pd.isna(b) or b <= 0 or a < 0:
        return 0.0
    ret = a / b - 1
    return max(-max_ret, min(max_ret, ret))


def _safe_ret_value(r: float, max_ret: float = 0.5) -> float:
    """处理收益序列的单个值 (NaN-safe, 限制极端值).

    与 _safe_return 不同: 这是直接的收益率值, 不需要 a/b-1.
    """
    if pd.isna(r):
        return 0.0
    return max(-max_ret, min(max_ret, r))


def _performance_metrics(nav: pd.Series, freq: str = 'W') -> dict:
    """计算业绩指标 (复用 v9 backtest.compute_metrics)."""
    n = len(nav)
    if n < 2:
        return {}

    freq_map = {"D": 252, "W": 52, "M": 12}
    periods = freq_map.get(freq, 52)

    ret = nav.pct_change().dropna()
    if len(ret) < 2:
        return {"ann_return": 0.0}

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = n / periods
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = ret.std() * np.sqrt(periods)
    sharpe = (ret.mean() * periods) / ann_vol if ann_vol > 0 else 0

    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    max_dd = float(drawdown.min())

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

    win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0

    return {
        "ann_return": float(ann_ret),
        "total_return": float(total_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "final_nav": float(nav.iloc[-1]),
        "n_periods": int(n),
    }


def run_v11_backtest(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    ohlcv_df: pd.DataFrame | None = None,
    cfg: V11Config | None = None,
) -> V11BacktestResult:
    """v11 完整回测.

    参数:
        returns_df: (T, N) ETF 周收益
        macro_df: (T, K) 宏观因子 (Layer 1 可选)
        ohlcv_df: (T, N*5) OHLCV 数据 (ACT-1 需要)
        cfg: V11Config

    返回:
        V11BacktestResult: NAV + 权重 + 指标
    """
    cfg = cfg or V11Config()
    returns_df = returns_df.fillna(0).replace([np.inf, -np.inf], 0)

    # === Step 1: 生成权重时序 ===
    strategy = V11Strategy(cfg)
    weights_rebal = strategy.run(returns_df, macro_df, ohlcv_df)

    # === Step 2: 计算 NAV ===
    rebal_dates = weights_rebal.index.tolist()
    dates = returns_df.index
    nav = np.ones(len(dates))

    weights_today = pd.Series(0.0, index=returns_df.columns)
    last_rebal_idx = -1

    for i, date in enumerate(dates):
        is_rebal = date in rebal_dates

        if is_rebal:
            # 调仓日: 先扣除成本, 再计算收益
            if last_rebal_idx >= 0:
                # 计算换手率
                w_new = weights_rebal.loc[date]
                turnover = 0.5 * (w_new - weights_today).abs().sum()
                cost = turnover * cfg.cost_bps / 10000
                nav[i] = nav[i - 1] * (1 - cost)
            elif i > 0:
                nav[i] = nav[i - 1]

            weights_today = weights_rebal.loc[date]
            last_rebal_idx = i

            # 计算当日收益 (调仓日, 直接用收益率序列)
            if i > 0:
                port_ret = 0.0
                for code in weights_today.index:
                    if code in returns_df.columns:
                        r = returns_df[code].iloc[i]
                        port_ret += weights_today[code] * _safe_ret_value(r)
                nav[i] = nav[i] * (1 + port_ret)
            else:
                nav[i] = 1.0
        else:
            # 非调仓日: 用前一日权重累积收益
            if i > 0:
                port_ret = 0.0
                for code in weights_today.index:
                    if code in returns_df.columns:
                        r = returns_df[code].iloc[i]
                        port_ret += weights_today[code] * _safe_ret_value(r)
                nav[i] = nav[i - 1] * (1 + port_ret)
            else:
                nav[i] = 1.0

        # 防止 NAV 变负
        if nav[i] < 0:
            nav[i] = 0.0

    nav_series = pd.Series(nav, index=dates, name='v11_nav')

    # === Step 3: 业绩指标 ===
    metrics = _performance_metrics(nav_series, freq='W')

    return V11BacktestResult(
        nav=nav_series,
        weights=weights_rebal,
        macro_score=strategy.macro_score,
        regime_state=strategy.regime_state,
        bear_prob=strategy.bear_prob,
        position_size=strategy.position_size,
        metrics=metrics,
    )


__all__ = ["run_v11_backtest", "V11BacktestResult"]
