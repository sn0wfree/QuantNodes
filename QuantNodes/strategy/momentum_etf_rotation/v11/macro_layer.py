# coding=utf-8
"""Layer 1: 宏观择时 (Macro Regime).

来源: v9 factor_galaxy (熵权法) + v7 tvpr_estimator (TV-PR 时变β, 可选)

功能:
    1. 5 宏观因子 z-score (growth/inflation/credit/fx/rate)
    2. 熵权法合成综合得分 macro_score ∈ [-1, +1]
    3. TV-PR 时变β (可选, 默认开启) → macro_signal_tvpr
    4. 综合: macro_signal = (1-w) * entropy_score + w * tvpr_score
    5. regime_state: bull/neutral/bear (阈值 ±0.5)

输出:
    macro_score: Series (T,) ∈ [-1, +1]
    regime_state: Series (T,) ∈ {'bull', 'neutral', 'bear'}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v11 import MacroLayerConfig


def _safe_entropy_weight(data: pd.DataFrame, window: int = 104) -> dict:
    """熵权法: 信息熵越小, 权重越大.

    复用 v9 factor_galaxy.entropy_weight() 逻辑.
    """
    if len(data) < window:
        return {col: 1.0 / len(data.columns) for col in data.columns}

    recent = data.iloc[-window:].dropna()
    if recent.empty:
        return {col: 1.0 / len(data.columns) for col in data.columns}

    n = len(recent)
    weights = {}
    for col in recent.columns:
        x = recent[col].abs()
        if x.sum() == 0:
            weights[col] = 0.0
            continue
        p = x / x.sum()
        entropy = -(p * np.log(p + 1e-10)).sum() / np.log(n)
        weights[col] = 1 - entropy

    total = sum(weights.values())
    if total == 0:
        return {col: 1.0 / len(data.columns) for col in data.columns}
    return {k: v / total for k, v in weights.items()}


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 z-score 标准化."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / (std + 1e-10)).clip(-3, 3)


def compute_macro_z_scores(
    macro_df: pd.DataFrame,
    cfg: MacroLayerConfig,
) -> pd.DataFrame:
    """计算 5 宏观因子 z-score.

    参数:
        macro_df: (T, K) 宏观因子 DataFrame
        cfg: MacroLayerConfig

    返回:
        macro_z: (T, K) z-score DataFrame, 列名同 macro_df
    """
    out = pd.DataFrame(index=macro_df.index)
    for col in cfg.factor_signs.keys():
        if col not in macro_df.columns:
            continue
        sign = cfg.factor_signs[col]
        if sign == 0:
            continue
        z = _rolling_zscore(macro_df[col], cfg.zscore_window)
        out[col] = sign * z
    return out.fillna(0)


def compute_macro_score_entropy(
    macro_z: pd.DataFrame,
    cfg: MacroLayerConfig,
) -> pd.Series:
    """熵权法合成 macro_score."""
    weights = _safe_entropy_weight(macro_z, window=cfg.entropy_window)
    score = pd.Series(0.0, index=macro_z.index)
    for col in macro_z.columns:
        w = weights.get(col, 0)
        score = score + w * macro_z[col]
    return score.clip(-2, 2)


def compute_macro_score_tvpr(
    etf_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    cfg: MacroLayerConfig,
) -> pd.Series:
    """TV-PR 时变β → macro_signal.

    复用 v7 tvpr_estimator.expanding_window_tvpr.
    注意: 计算较慢, 如果 use_tvpr=False 则跳过.
    """
    if not cfg.use_tvpr:
        return pd.Series(0.0, index=etf_returns.index)

    from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
        expanding_window_tvpr,
    )

    # 对齐
    common = etf_returns.index.intersection(macro_df.index)
    Y = etf_returns.loc[common].fillna(0).values
    X = macro_df.loc[common].fillna(0).values

    if len(Y) < cfg.tvpr_min_history + 10:
        return pd.Series(0.0, index=common)

    try:
        beta = expanding_window_tvpr(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            min_history=cfg.tvpr_min_history,
        )
        # β 的横截面均值 → 时间序列 → z-score
        beta_mean = beta.mean(axis=1)
        beta_z = _rolling_zscore(pd.Series(beta_mean, index=common), cfg.zscore_window)
        return beta_z.clip(-2, 2).fillna(0)
    except Exception:
        return pd.Series(0.0, index=common)


def compute_macro_signal(
    macro_df: pd.DataFrame,
    etf_returns: pd.DataFrame | None,
    cfg: MacroLayerConfig,
) -> tuple[pd.Series, pd.Series]:
    """Layer 1 主入口: 计算 macro_score 和 regime_state.

    参数:
        macro_df: (T, K) 宏观因子 DataFrame (列名见 cfg.factor_signs)
        etf_returns: (T, N) ETF 收益 DataFrame (TV-PR 必需)
        cfg: MacroLayerConfig

    返回:
        macro_score: (T,) 综合得分 ∈ [-1, +1]
        regime_state: (T,) ∈ {'bull', 'neutral', 'bear'}
    """
    if not cfg.enabled or macro_df is None or macro_df.empty:
        # 默认中性
        idx = etf_returns.index if etf_returns is not None else macro_df.index
        return pd.Series(0.0, index=idx), pd.Series('neutral', index=idx)

    # 1. 5 宏观因子 z-score
    macro_z = compute_macro_z_scores(macro_df, cfg)

    # 2. 熵权法综合得分
    entropy_score = compute_macro_score_entropy(macro_z, cfg)

    # 3. TV-PR (可选)
    if cfg.use_tvpr and etf_returns is not None:
        tvpr_score = compute_macro_score_tvpr(etf_returns, macro_df, cfg)
        # 对齐 index
        tvpr_score = tvpr_score.reindex(entropy_score.index, method='ffill').fillna(0)
        macro_score = (1 - cfg.tvpr_weight) * entropy_score + cfg.tvpr_weight * tvpr_score
    else:
        macro_score = entropy_score

    macro_score = macro_score.clip(-1.5, 1.5)

    # 4. Regime 分类
    regime_state = pd.Series('neutral', index=macro_score.index)
    regime_state[macro_score > cfg.bull_threshold] = 'bull'
    regime_state[macro_score < cfg.bear_threshold] = 'bear'

    return macro_score, regime_state


# ============================================================
# 类封装 (便于 v10_strategy 统一调用)
# ============================================================
class MacroLayer:
    """Layer 1 宏观择时封装."""

    def __init__(self, cfg: MacroLayerConfig | None = None):
        self.cfg = cfg or MacroLayerConfig()
        self.macro_score: pd.Series | None = None
        self.regime_state: pd.Series | None = None

    def fit(self, macro_df: pd.DataFrame, etf_returns: pd.DataFrame | None = None) -> "MacroLayer":
        """计算 macro_score 和 regime_state."""
        self.macro_score, self.regime_state = compute_macro_signal(
            macro_df, etf_returns, self.cfg,
        )
        return self

    def get_score(self, date: pd.Timestamp) -> float:
        """获取指定日期的 macro_score."""
        if self.macro_score is None:
            return 0.0
        if date not in self.macro_score.index:
            return 0.0
        return float(self.macro_score.loc[date])

    def get_regime(self, date: pd.Timestamp) -> str:
        """获取指定日期的 regime."""
        if self.regime_state is None:
            return 'neutral'
        if date not in self.regime_state.index:
            return 'neutral'
        return str(self.regime_state.loc[date])
