# coding=utf-8
"""v10 回测引擎 — 完整回测 + 业绩指标.

基于 v4 multi_strategy_v4.run_v4_backtest() 框架 + v9 backtest.run_backtest() 复用.

功能:
    1. 调仓日: 用 V10Strategy 生成的权重
    2. 非调仓日: 用前一日权重累积
    3. 调仓成本: 5bp 单边
    4. NAV 计算: 含成本
    5. 业绩指标: 委托给 common.metrics.compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common.metrics import compute_metrics

from .config_v10 import V10Config
from .v10_strategy import V10Strategy


@dataclass
class V10BacktestResult:
    """v10 回测结果."""
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
    """计算业绩指标 (委托给 common.metrics.compute_metrics).

    兼容 freq='W' / 'D' / 'M' 字符串参数.
    """
    freq_map = {"D": 252, "W": 52, "M": 12}
    periods = freq_map.get(freq, 52)

    base = compute_metrics(nav, freq=periods)

    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1) if not nav.empty else 0.0

    return {
        "ann_return": base["AnnRet"],
        "total_return": total_ret,
        "ann_vol": base["Vol"],
        "sharpe": base["Sharpe"],
        "max_drawdown": base["MaxDD"],
        "calmar": base["Calmar"],
        "win_rate": base["WinRate"],
        "final_nav": float(nav.iloc[-1]) if not nav.empty else 0.0,
        "n_periods": int(len(nav)),
    }


def run_v10_backtest(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    cfg: V10Config | None = None,
) -> V10BacktestResult:
    """v10 完整回测.

    参数:
        returns_df: (T, N) ETF 周收益
        macro_df: (T, K) 宏观因子 (Layer 1 可选)
        cfg: V10Config

    返回:
        V10BacktestResult: NAV + 权重 + 指标
    """
    cfg = cfg or V10Config()
    returns_df = returns_df.fillna(0).replace([np.inf, -np.inf], 0)

    # === Step 1: 生成权重时序 ===
    strategy = V10Strategy(cfg)
    weights_rebal = strategy.run(returns_df, macro_df)

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

    nav_series = pd.Series(nav, index=dates, name='v10_nav')

    # === Step 3: 业绩指标 ===
    metrics = _performance_metrics(nav_series, freq='W')

    return V10BacktestResult(
        nav=nav_series,
        weights=weights_rebal,
        macro_score=strategy.macro_score,
        regime_state=strategy.regime_state,
        bear_prob=strategy.bear_prob,
        position_size=strategy.position_size,
        metrics=metrics,
    )


__all__ = ["run_v10_backtest", "V10BacktestResult"]