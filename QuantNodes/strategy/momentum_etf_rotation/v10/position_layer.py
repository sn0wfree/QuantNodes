# coding=utf-8
"""Layer 4: 动态仓位 (Dynamic Position).

来源: v9 银河方案 (Brinson 归因 71% alpha) + Jump Model 双控

公式:
    z_score = w_macro × macro_score
            + w_sector × sector_score
            + w_style × style_weight_weighted_mean

    position_size = (pos_intercept - pos_z_coef × z_score).clip(pos_min, pos_max)
    position_size *= (1 - bear_prob × bear_adjustment_coef)  # Jump Model

输出:
    position_size: Series (T,) ∈ [pos_min, pos_max]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v10 import PositionLayerConfig


def compute_z_score(
    macro_score: pd.Series | None,
    sector_tilt: pd.DataFrame | None,
    style_weights: pd.DataFrame | None,
    cfg: PositionLayerConfig,
) -> pd.Series:
    """合成 z_score.

    参数:
        macro_score: (T,) Layer 1 输出
        sector_tilt: (T_rebal, N) Layer 2A 输出
        style_weights: (T_rebal, 6) Layer 2B 输出
        cfg: PositionLayerConfig

    返回:
        z_score: (T,) Series
    """
    z = pd.Series(0.0, index=macro_score.index if macro_score is not None else sector_tilt.index)
    weights = cfg.z_score_weights

    # 1. Macro
    if macro_score is not None and weights.get('macro', 0) > 0:
        z = z + weights['macro'] * macro_score.reindex(z.index, method='ffill').fillna(0)

    # 2. Sector (横截面均值 → 时序)
    if sector_tilt is not None and not sector_tilt.empty and weights.get('sector', 0) > 0:
        # sector_tilt 中 >1 是 Top-K (sector_mult=5), <1 是 floor (0.5)
        # 归一化到 [-1, +1]
        sector_mean = sector_tilt.mean(axis=1)
        # sector_mean ∈ [0.5, 5.0] (默认), 标准化
        sector_z = (sector_mean - 1.0) / 2.0  # 粗略: 1.0 → 0, 5.0 → +2
        sector_z = sector_z.reindex(z.index, method='ffill').fillna(0)
        z = z + weights['sector'] * sector_z

    # 3. Style (加权均值 → 时序)
    if style_weights is not None and not style_weights.empty and weights.get('style', 0) > 0:
        # style_weights 列名: momentum, value, reversal, quality, size, low_vol
        # 等权合成 → 越偏 momentum/quality 越好, 越偏 value/reversal 越差
        # 简化: 把 momentum + quality 视为正向, value + reversal 视为反向
        style_score = pd.Series(0.0, index=style_weights.index)
        for col in ['momentum', 'quality']:
            if col in style_weights.columns:
                style_score += style_weights[col]
        for col in ['value', 'reversal', 'low_vol']:
            if col in style_weights.columns:
                style_score -= style_weights[col]
        style_score = style_score.reindex(z.index, method='ffill').fillna(0)
        z = z + weights['style'] * style_score

    return z


def compute_dynamic_position(
    z_score: pd.Series,
    bear_prob: pd.Series | None,
    cfg: PositionLayerConfig,
) -> pd.Series:
    """计算动态仓位时序.

    参数:
        z_score: (T,) z-score 综合信号
        bear_prob: (T,) Jump Model bear 概率
        cfg: PositionLayerConfig

    返回:
        position_size: (T,) Series ∈ [pos_min, pos_max]
    """
    if not cfg.enabled:
        return pd.Series(1.0, index=z_score.index)

    # 1. z_score → position_size (沿用 v9 公式)
    pos = cfg.pos_intercept - cfg.pos_z_coef * z_score
    pos = pos.clip(cfg.pos_min, cfg.pos_max)

    # 2. Jump Model bear_prob 调整
    if cfg.use_bear_prob_adjustment and bear_prob is not None:
        bear_aligned = bear_prob.reindex(pos.index, method='ffill').fillna(0)
        pos = pos * (1 - cfg.bear_prob_adjustment_coef * bear_aligned)
        pos = pos.clip(cfg.pos_min, cfg.pos_max)

    return pos


def compute_position_size(
    macro_score: pd.Series | None,
    sector_tilt: pd.DataFrame | None,
    style_weights: pd.DataFrame | None,
    bear_prob: pd.Series | None,
    cfg: PositionLayerConfig,
) -> pd.Series:
    """Layer 4 主入口: 计算动态仓位时序."""
    z = compute_z_score(macro_score, sector_tilt, style_weights, cfg)
    pos = compute_dynamic_position(z, bear_prob, cfg)
    return pos


# ============================================================
# 类封装
# ============================================================
class PositionLayer:
    """Layer 4 动态仓位封装."""

    def __init__(self, cfg: PositionLayerConfig | None = None):
        self.cfg = cfg or PositionLayerConfig()
        self.position_size: pd.Series | None = None
        self.z_score: pd.Series | None = None

    def fit(self, macro_score: pd.Series | None = None,
            sector_tilt: pd.DataFrame | None = None,
            style_weights: pd.DataFrame | None = None,
            bear_prob: pd.Series | None = None) -> "PositionLayer":
        """计算动态仓位时序."""
        self.z_score = compute_z_score(macro_score, sector_tilt, style_weights, self.cfg)
        self.position_size = compute_dynamic_position(self.z_score, bear_prob, self.cfg)
        return self

    def get_position(self, date: pd.Timestamp) -> float:
        """获取指定日期的仓位."""
        if self.position_size is None or date not in self.position_size.index:
            return 1.0
        return float(self.position_size.loc[date])