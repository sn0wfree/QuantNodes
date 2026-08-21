# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_factor_groups.py

v6.3 多因子体系 - 组内合成层

将12个因子按组别加权合成，再跨组合成最终仓位。

默认权重:
    组间: 动量30%, 宏观25%, 库存20%, 期限15%, 波动5%, 跨资产5%
    组内: 等权 (可配置)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 默认组间权重 (基于单因子Sharpe优化)
DEFAULT_GROUP_WEIGHTS = {
    "momentum": 0.15,      # mom_2w(0.50) + mom_13w(0.50)
    "macro": 0.45,         # dxy_z(0.66) + pmi(0.65) + industry_dz(0.53) - 最强组
    "inventory": 0.10,     # lme_z(0.42) + quadrant_pos(0.53)
    "term_structure": 0.10, # carry(0.50) + curvature(0.52)
    "volatility": 0.10,    # vol_13w_z(0.58) + vol_pct(0.42)
    "cross_asset": 0.10,   # cu_crude(0.54)
}

# 默认组内权重 (基于单因子Sharpe优化)
DEFAULT_FACTOR_WEIGHTS = {
    "momentum": {"mom_2w": 0.5, "mom_13w": 0.5},
    "macro": {"dxy_z": 0.50, "pmi": 0.40, "industry_dz": 0.10},  # dxy和pmi最强
    "inventory": {"lme_z": 0.4, "quadrant_pos": 0.6},
    "term_structure": {"carry": 0.5, "curvature": 0.5},
    "volatility": {"vol_13w_z": 0.7, "vol_pct": 0.3},
    "cross_asset": {"cu_crude": 1.0},
}

# 因子方向: 正=做多, 负=做空
FACTOR_DIRECTION = {
    "mom_2w": 1,
    "mom_13w": 1,
    "dxy_z": -1,    # 美元弱=做多
    "pmi": 1,
    "industry_dz": 1,
    "lme_z": -1,    # 库存低=做多
    "quadrant_pos": 1,
    "carry": 1,     # backwardation=做多
    "curvature": 1,
    "vol_13w_z": 1,
    "vol_pct": -1,  # 低波=做多
    "cu_crude": 1,
}


def normalize_factor(factor: pd.Series, direction: int = 1) -> pd.Series:
    """归一化因子到[0, 1]区间，并调整方向.

    输入: 原始因子值 (任意范围)
    输出: [0, 1]区间, 1=做多, 0=空仓

    使用排名归一化，对异常值更鲁棒。
    """
    # 排名归一化到[0, 1]
    ranked = factor.rank(pct=True).fillna(0.5)

    # 调整方向
    if direction < 0:
        ranked = 1 - ranked

    return ranked.clip(0, 1)


def combine_group(factors_df: pd.DataFrame, group_name: str,
                  factor_weights: dict) -> pd.Series:
    """组内加权合成.

    参数:
        factors_df: 因子DataFrame
        group_name: 组名
        factor_weights: 组内因子权重 {factor_name: weight}

    返回: 组合成仓位 [0, 1]
    """
    group_positions = []
    weights = []

    for fname, weight in factor_weights.items():
        if fname in factors_df.columns:
            direction = FACTOR_DIRECTION.get(fname, 1)
            # 排名归一化
            pos = normalize_factor(factors_df[fname], direction)
            group_positions.append(pos.fillna(0.5))
            weights.append(weight)

    if not group_positions:
        return pd.Series(0.5, index=factors_df.index)

    # 加权平均
    total_weight = sum(weights)
    result = sum(p * w for p, w in zip(group_positions, weights)) / total_weight

    return result.clip(0, 1)


def combine_all_groups(factors_df: pd.DataFrame,
                       group_weights: dict = None,
                       factor_weights: dict = None) -> pd.Series:
    """跨组合成最终仓位.

    参数:
        factors_df: 因子DataFrame
        group_weights: 组间权重 {group_name: weight}
        factor_weights: 组内因子权重 {group_name: {factor_name: weight}}

    返回: 最终仓位 [0, 1]
    """
    if group_weights is None:
        group_weights = DEFAULT_GROUP_WEIGHTS
    if factor_weights is None:
        factor_weights = DEFAULT_FACTOR_WEIGHTS

    # 组内合成
    group_scores = {}
    for group_name, gw in group_weights.items():
        if group_name in factor_weights:
            group_scores[group_name] = combine_group(
                factors_df, group_name, factor_weights[group_name]
            )
        else:
            group_scores[group_name] = pd.Series(0.5, index=factors_df.index)

    # 跨组合成: 加权平均
    total_gw = sum(group_weights.values())
    result = sum(group_scores[g] * gw for g, gw in group_weights.items()) / total_gw

    return result.clip(0, 1)


__all__ = [
    "combine_group",
    "combine_all_groups",
    "normalize_factor",
    "DEFAULT_GROUP_WEIGHTS",
    "DEFAULT_FACTOR_WEIGHTS",
    "FACTOR_DIRECTION",
]
