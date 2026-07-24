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

import numpy as np
import pandas as pd

from .config_v10 import RiskLayerConfig


def _resample_to_daily(returns_df: pd.DataFrame) -> pd.DataFrame:
    """周频 → 日频 (Jump Model 需要).

    将周收益扩展为日收益 (假设均匀分布):
        daily_ret = (1 + weekly_ret) ** (1/5) - 1
    """
    # 简单近似: 周收益 / 5 (不能用幂分解因为会重复计收益)
    daily_ret = returns_df / 5
    # 扩展到日频索引
    daily_index = pd.date_range(
        start=returns_df.index.min(),
        end=returns_df.index.max() + pd.Timedelta(days=6),
        freq='B',
    )
    daily_df = pd.DataFrame(index=daily_index[:len(returns_df) * 5], columns=returns_df.columns)
    # 按周填充
    for i, week_end in enumerate(returns_df.index):
        if i * 5 >= len(daily_df):
            break
        daily_df.iloc[i * 5: (i + 1) * 5] = returns_df.loc[week_end].values
    return daily_df.fillna(0).infer_objects()


def _compute_bear_probability_from_states(
    states: pd.Series,
    window: int = 60,
) -> pd.Series:
    """从状态序列计算滚动 bear 概率."""
    # states: 0=bull, 1=bear
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

    # 简化版: 用日频收益 (从周频扩展)
    # 注: 完整版需要日频数据, 这里用周频代理
    # 取等权市场组合作为 Jump Model 输入
    market_returns = returns_df.mean(axis=1)

    try:
        # 复用 v8 jump_model_periodic_retrain (支持 equity/bond/commodity)
        states = jump_model_periodic_retrain(
            returns=market_returns,
            asset_type=cfg.asset_type,
            jump_penalty=cfg.jump_penalty,
            train_window=cfg.train_window if cfg.train_window else 200,  # 周 → 周窗口
            retrain_every=cfg.retrain_every if cfg.retrain_every else 13,  # 周
            n_iter=cfg.n_iter,
            n_restarts=cfg.n_restarts,
            show_progress=False,
        )
    except Exception as e:
        # Jump Model 失败时返回中性
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