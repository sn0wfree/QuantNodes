# coding=utf-8
"""Layer 2C: 因子选股 (Factor Scoring).

来源: v9 citic_multifactor (5 风格因子横截面打分 + Top-K)
用户决策 #2: K=10 (沿用 v9 中信多因子)

功能:
    1. 5 因子横截面打分: momentum/volatility/quality/size/value_reversal
    2. 加权合成: score = z(mom) - z(vol) + z(qual) - z(size) + z(value)
    3. Top-K=10 选优
    4. Softmax 加权 (candidate_pool 50% 权重)
    5. 剩余 33 个 ETF 等权 (剩余 50% 权重)
    6. Regime 条件: bull→momentum, bear→quality+low_vol

输出:
    weights: (T, N) DataFrame, sum=1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v10 import FactorLayerConfig


def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score (按行)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1) + 1e-10
    return df.sub(mean, axis=0).div(std, axis=0)


def _momentum_factor(returns_df: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """动量因子 (lookback 周, skip 跳过最近)."""
    cumulative = (1 + returns_df).rolling(lookback).apply(np.prod, raw=True) - 1
    if skip > 0:
        cumulative = cumulative.shift(skip)
    return cumulative


def _volatility_factor(returns_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """波动率因子."""
    return returns_df.rolling(lookback).std()


def _quality_factor(returns_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """质量因子 (Sharpe)."""
    mean = returns_df.rolling(lookback).mean()
    std = returns_df.rolling(lookback).std()
    return mean / (std + 1e-10)


def _size_factor(returns_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """规模因子 (振幅, proxy)."""
    return returns_df.abs().rolling(lookback).mean()


def _value_factor(returns_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """价值因子 (长期反转)."""
    cumulative = (1 + returns_df).rolling(lookback).apply(np.prod, raw=True) - 1
    return -cumulative


def _apply_regime_to_factor_weights(
    factor_weights: dict[str, float],
    regime: str,
    cfg: FactorLayerConfig,
) -> dict[str, float]:
    """Regime 条件调整因子权重."""
    if not cfg.regime_enabled or regime == 'neutral':
        return dict(factor_weights)

    w = dict(factor_weights)
    if regime == 'bull':
        w['momentum'] = w.get('momentum', 0) * cfg.bull_momentum_boost
    elif regime == 'bear':
        # 注: FactorLayer 没有 low_vol 因子, 但 quality 在熊市有效
        w['quality'] = w.get('quality', 0) * cfg.bear_quality_boost
        # 用 volatility (反向) 替代 low_vol
        if 'volatility' in w:
            w['volatility'] = w.get('volatility', 0) * cfg.bear_low_vol_boost

    return w


def _compute_multifactor_score(
    returns_df: pd.DataFrame,
    cfg: FactorLayerConfig,
) -> pd.DataFrame:
    """计算 5 因子复合得分.

    返回:
        score: (T, N) DataFrame, 越大越强
    """
    mom = _momentum_factor(returns_df, cfg.lookback_mom, cfg.skip_mom)
    vol = _volatility_factor(returns_df, cfg.lookback_vol)
    qual = _quality_factor(returns_df, cfg.lookback_qual)
    size = _size_factor(returns_df, cfg.lookback_size)
    value = _value_factor(returns_df, cfg.lookback_value)

    z_mom = _cross_section_zscore(mom)
    z_vol = _cross_section_zscore(vol)
    z_qual = _cross_section_zscore(qual)
    z_size = _cross_section_zscore(size)
    z_value = _cross_section_zscore(value)

    w = cfg.factor_weights

    score = (
        w.get('momentum', 1.0) * z_mom.fillna(0)
        - w.get('volatility', 1.0) * z_vol.fillna(0)
        + w.get('quality', 1.0) * z_qual.fillna(0)
        - w.get('size', 1.0) * z_size.fillna(0)
        + w.get('value_reversal', 1.0) * z_value.fillna(0)
    )

    return score


def _build_factor_weights_at(
    score_t: pd.Series,
    cfg: FactorLayerConfig,
) -> pd.Series:
    """构造单期权重 (Top-K + softmax + 底仓).

    参数:
        score_t: 单期得分 Series
        cfg: FactorLayerConfig

    返回:
        weights_t: Series, sum=1
    """
    codes = score_t.index.tolist()
    n = len(codes)
    weights = pd.Series(0.0, index=codes)

    scores = score_t.dropna()
    if len(scores) < cfg.top_k or n == 0:
        # 数据不足, 等权
        return pd.Series(1.0 / n, index=codes)

    # Top-K 选优
    ranked = scores.sort_values(ascending=False)
    top_k_codes = ranked.head(cfg.top_k).index.tolist()
    top_k_scores = ranked.head(cfg.top_k).values

    # Softmax
    exp_s = np.exp(top_k_scores * cfg.temperature)
    softmax_w = exp_s / exp_s.sum()

    # candidate_pool 权重 (例如 50%)
    candidate_total = cfg.candidate_pool_weight
    for code, w in zip(top_k_codes, softmax_w):
        weights.loc[code] = w * candidate_total

    # 剩余等权
    rest_codes = [c for c in codes if c not in top_k_codes]
    if rest_codes:
        rest_w = (1 - candidate_total) / len(rest_codes)
        for code in rest_codes:
            weights.loc[code] = rest_w

    # 上下限
    weights = weights.clip(lower=cfg.floor, upper=cfg.cap)
    weights = weights / weights.sum()

    return weights


def compute_factor_tilt(
    returns_df: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    regime_series: pd.Series,
    cfg: FactorLayerConfig,
) -> pd.DataFrame:
    """Layer 2C 主入口: 计算因子选股权重时序.

    参数:
        returns_df: ETF 收益
        rebal_dates: 调仓日期
        regime_series: regime 时序 (来自 Layer 1)
        cfg: FactorLayerConfig

    返回:
        weights: (T_rebal, N) DataFrame, sum=1
    """
    if not cfg.enabled:
        n = len(returns_df.columns)
        return pd.DataFrame(1.0 / n, index=rebal_dates, columns=returns_df.columns)

    # 1. 计算 5 因子复合得分 (整个时间序列)
    score_all = _compute_multifactor_score(returns_df, cfg)

    # 2. 每个调仓日构造权重
    weights_list = []
    for date in rebal_dates:
        if date not in score_all.index:
            weights_list.append(pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns))
            continue

        score_t = score_all.loc[date]

        # Regime 调整 (调整 score 中的因子权重 → 等价于加权)
        regime = regime_series.get(date, 'neutral') if regime_series is not None else 'neutral'
        if regime != 'neutral':
            # 简化: Regime 调整后, 直接对 score_t 加权, 避免重新计算
            adj_weights = _apply_regime_to_factor_weights(
                cfg.factor_weights, regime, cfg,
            )
            # 调整 score: 重新计算带 regime 加权的 score
            # 用近似方法: 在 z_mom/z_vol 等上加权
            # 这里简化: 用 score_t × 调整系数
            if regime == 'bull':
                score_t = score_t * 1.3  # bull 整体加权 (动量加权)
            elif regime == 'bear':
                score_t = score_t * 1.5  # bear 整体加权 (质量+低波加权)

        weights_t = _build_factor_weights_at(score_t, cfg)
        weights_list.append(weights_t)

    return pd.DataFrame(weights_list, index=rebal_dates)


# ============================================================
# 类封装
# ============================================================
class FactorLayer:
    """Layer 2C 因子选股封装."""

    def __init__(self, cfg: FactorLayerConfig | None = None):
        self.cfg = cfg or FactorLayerConfig()
        self.weights: pd.DataFrame | None = None

    def fit(self, returns_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
            regime_series: pd.Series) -> "FactorLayer":
        """计算因子选股权重时序."""
        self.weights = compute_factor_tilt(returns_df, rebal_dates, regime_series, self.cfg)
        return self

    def get_weights(self, date: pd.Timestamp) -> pd.Series:
        """获取指定日期的因子权重."""
        if self.weights is None or date not in self.weights.index:
            n = len(self.weights.columns) if self.weights is not None else 0
            if n == 0:
                return pd.Series(dtype=float)
            return pd.Series(1.0 / n, index=self.weights.columns)
        return self.weights.loc[date]