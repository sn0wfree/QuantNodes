# coding=utf-8
"""Layer 3: 风险控制 (Risk Management) — Jump Model.

来源: v8 jump_model.jump_model_periodic_retrain (无未来函数版)

功能:
    1. 在 ETF 池上跑 Jump Model, 输出 bull/bear 状态
    2. 计算滚动 bear 概率 (60 周窗口)
    3. 输出 bear_probability ∈ [0, 1]

输出:
    bear_prob: Series (T,) ∈ [0, 1]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config_v10 import RiskLayerConfig

REPO = Path(__file__).resolve().parents[4]
REAL_DIR = REPO / "data" / "real" / "per_etf"


def _load_daily_nav(codes: list[str]) -> pd.DataFrame:
    """直接加载日频 NAV 数据."""
    navs = {}
    for code in codes:
        path = REAL_DIR / f"{code}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if 'close' in df.columns:
                navs[code] = df['close']
    if not navs:
        return pd.DataFrame()
    return pd.DataFrame(navs)


def _load_daily_returns(codes: list[str]) -> pd.DataFrame:
    """加载日频收益数据."""
    nav = _load_daily_nav(codes)
    if nav.empty:
        return pd.DataFrame()
    return nav.pct_change().fillna(0)


def _compute_bear_probability_from_states(
    states: pd.Series,
    window: int = 60,
) -> pd.Series:
    """从状态序列计算滚动 bear 概率."""
    bear_indicator = (states == 1).astype(float)
    bear_prob = bear_indicator.rolling(window, min_periods=window // 2).mean()
    return bear_prob.fillna(0)


def compute_bear_probability(
    returns_df: pd.DataFrame,
    cfg: RiskLayerConfig,
) -> pd.Series:
    """Layer 3 主入口: 计算 bear_probability 时序.

    参数:
        returns_df: (T_weekly, N) 周频 ETF 收益
        cfg: RiskLayerConfig

    返回:
        bear_prob: (T_weekly,) Series, ∈ [0, 1]
    """
    if not cfg.enabled:
        return pd.Series(0.0, index=returns_df.index)

    from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
        jump_model_periodic_retrain,
    )

    # 直接加载日频收益数据
    codes = returns_df.columns.tolist()
    daily_returns = _load_daily_returns(codes)

    if daily_returns.empty:
        # 回退到周频数据
        market_returns = returns_df.mean(axis=1)
    else:
        # 使用日频数据的等权组合收益
        market_returns = daily_returns.mean(axis=1)

    try:
        states = jump_model_periodic_retrain(
            returns=market_returns,
            asset_type=cfg.asset_type,
            jump_penalty=cfg.jump_penalty,
            train_window=cfg.train_window if cfg.train_window else 1000,  # 1000 天 ≈ 200 周
            retrain_every=cfg.retrain_every if cfg.retrain_every else 65,  # 65 天 ≈ 13 周
            n_iter=cfg.n_iter,
            n_restarts=cfg.n_restarts,
            show_progress=False,
        )
    except Exception:
        return pd.Series(0.0, index=returns_df.index)

    # 对齐到周频
    states = states.reindex(returns_df.index, method='ffill').fillna(0)

    # 计算滚动 bear 概率
    bear_prob = _compute_bear_probability_from_states(states, window=cfg.bear_prob_window)

    return bear_prob


# ============================================================
# 类封装
# ============================================================
class RiskLayer:
    """Layer 3 风险控制封装."""

    def __init__(self, cfg: RiskLayerConfig | None = None):
        self.cfg = cfg or RiskLayerConfig()
        self.bear_prob: pd.Series | None = None

    def fit(self, returns_df: pd.DataFrame) -> "RiskLayer":
        """计算 bear_probability 时序."""
        self.bear_prob = compute_bear_probability(returns_df, self.cfg)
        return self

    def get_bear_prob(self, date: pd.Timestamp) -> float:
        """获取指定日期的 bear 概率."""
        if self.bear_prob is None or date not in self.bear_prob.index:
            return 0.0
        return float(self.bear_prob.loc[date])
