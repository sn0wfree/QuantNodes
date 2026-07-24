# coding=utf-8
"""v9 因子配置主入口 — 5 类宏观指标 + 银河因子配置.

5 类宏观指标映射 (用现有 17 因子代理):
    消费/内需: 宏观增长因子, 宏观通胀因子_生活端
    出口/外部: 宏观汇率因子, dxy_logret, cn_us_spread
    工业/生产: 宏观通胀因子_生产端
    信贷/金融: 信用利差因子, 无风险收益率, real_rate, 
               期限利差因子_债, 期限利差因子_股
    风险/情绪: vix, vix_rank20, tf_dummy, gold_oil_corr
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_galaxy import galaxy_factor_allocation


CATEGORY_MAPPING = {
    '消费/内需': ['宏观增长因子', '宏观通胀因子_生活端'],
    '出口/外部': ['宏观汇率因子', 'dxy_logret', 'cn_us_spread'],
    '工业/生产': ['宏观通胀因子_生产端'],
    '信贷/金融': [
        '信用利差因子', '无风险收益率', 'real_rate',
        '期限利差因子_债', '期限利差因子_股',
    ],
    '风险/情绪': ['vix', 'vix_rank20', 'tf_dummy', 'gold_oil_corr'],
}


def map_to_categories(macro_df: pd.DataFrame) -> pd.DataFrame:
    """将 17 因子映射到 5 类宏观指标 (类内等权平均).

    返回:
        category_macro: (T, 5) DataFrame, 列名为 5 类宏观指标
    """
    category_scores = {}
    for cat, factors in CATEGORY_MAPPING.items():
        available = [f for f in factors if f in macro_df.columns]
        if available:
            category_scores[cat] = macro_df[available].mean(axis=1)
    return pd.DataFrame(category_scores)


def run_factor_allocator(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    lookback_score: int = 104,
    lookback_beta: int = 52,
    target_budget: dict | None = None,
    floor: float = 0.02,
    cap: float = 0.20,
) -> tuple:
    """主入口: 17 因子 → 银河因子配置.

    银河证券方法: 不聚合成 5 类, 用 17 因子分别计算 β.
    这样能捕捉到本土股票对消费/增长因子的真实敏感度.

    参数:
        returns_df: (T, N) 资产收益
        macro_df: (T, 17) 17 宏观因子
        lookback_score: 熵权法窗口 (默认 104 周)
        lookback_beta: β 回归窗口 (默认 52 周)
        target_budget: 目标风险贡献
        floor/cap: 单资产权重上下限

    返回:
        weights: 周度权重
        factor_score: 综合得分
        betas: 因子敏感度
    """
    common_idx = returns_df.index.intersection(macro_df.index)
    returns_aligned = returns_df.loc[common_idx].fillna(0)
    macro_aligned = macro_df.loc[common_idx].fillna(0)

    weights, factor_score, betas = galaxy_factor_allocation(
        returns_df=returns_aligned,
        macro_indicators=macro_aligned,
        lookback_score=lookback_score,
        lookback_beta=lookback_beta,
        target_budget=target_budget,
        floor=floor, cap=cap,
    )
    return weights, factor_score, betas, macro_aligned


def get_category_exposures(weights: pd.DataFrame, 
                            cat_returns: pd.DataFrame) -> pd.DataFrame:
    """计算各宏观类别的风险贡献."""
    common = weights.index.intersection(cat_returns.index)
    w = weights.loc[common]
    r = cat_returns.loc[common]

    returns_contrib = pd.DataFrame(index=common)
    for cat in r.columns:
        returns_contrib[cat] = w.mean(axis=1) * r[cat]

    return returns_contrib