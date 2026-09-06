# coding: utf-8
"""60 月滚动 OLS Beta (CS 维度对齐) — P3 实装.

论文 §3.1 关键设计: 宏观/情绪不随品种变, 用 60 月滚动 OLS Beta
构造商品级 β_g,i / β_s,i, 实现 81→多维商品面板.

公式: beta_i = Cov(r_i, m_g, window) / Var(m_g, window)
月度截面, NaN 在前 window 月填充.

⚠️ 本项目数据 2018-2025 (7 年), 不严格沿用 60 月 (留前 60 月 NaN, 仅末段有值).
调整: 默认 window=36 (按 120 月滚动训练窗口的 30% 比例), 平衡数据可用性.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import BETA_ROLL_WINDOW

LOG = logging.getLogger(__name__)


def rolling_beta(returns: pd.DataFrame, macro: pd.DataFrame,
                 window: int | None = None) -> pd.DataFrame:
    """60 月滚动 OLS Beta.

    Args:
        returns: 商品收益 DataFrame, columns=product, index=MonthEnd.
        macro: 宏观变量 DataFrame, columns=feature, index=MonthEnd.
        window: 滚动窗口 (默认 config.BETA_ROLL_WINDOW=60).

    Returns:
        Beta DataFrame: index=trade_date, columns=MultiIndex (product, macro_feature).
        NaN 在前 (window-1) 月填充.
    """
    window = window or BETA_ROLL_WINDOW
    LOG.info(f"60 月滚动 Beta: {len(returns.columns)} 品种 × {len(macro.columns)} 宏观, "
             f"window={window}")
    common = returns.index.intersection(macro.index)
    if common.empty:
        LOG.warning("returns 和 macro 无共同日期")
        return pd.DataFrame()
    r = returns.loc[common].copy()
    m = macro.loc[common].copy()

    out_cols = {}
    for prod in r.columns:
        for feat in m.columns:
            y = r[prod]
            x = m[feat]
            beta = y.rolling(window).cov(x) / x.rolling(window).var()
            out_cols[(prod, feat)] = beta
    out = pd.DataFrame(out_cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["product", "feature"])
    return out


def build_commodity_panel(features_c: pd.DataFrame, features_m: pd.DataFrame,
                          returns: pd.DataFrame,
                          window: int | None = None) -> pd.DataFrame:
    """合成商品面板: Group C 原始 + (Group M × 商品级 Beta).

    Args:
        features_c: Group C 12 类变量, MultiIndex columns (product, feature).
        features_m: Group M 10 个宏观变量, columns=feature.
        returns: 商品月收益, columns=product (用于 rolling_beta).
        window: 滚动窗口 (默认 60).

    Returns:
        DataFrame: columns=MultiIndex (product, feature), index=MonthEnd.
    """
    LOG.info("build_commodity_panel: 计算 60 月滚动 Beta + 合成面板")
    beta = rolling_beta(returns, features_m, window=window)
    if beta.empty:
        return pd.DataFrame()

    # 透视 beta (product × macro_feature) → 列 (product_macro-feature) for each product
    panels = []
    for prod in features_c.columns.levels[0]:
        if prod in beta.columns.levels[0]:
            # 该商品的所有宏观 Beta 列
            macro_for_prod = beta[prod].copy()
            macro_for_prod.columns = pd.MultiIndex.from_product(
                [[prod], macro_for_prod.columns], names=["product", "feature"])
            panels.append(macro_for_prod)
        else:
            # 没有 beta (没收益), 全 NaN
            macro_for_prod = pd.DataFrame(
                np.nan, index=beta.index,
                columns=pd.MultiIndex.from_product(
                    [[prod], features_m.columns], names=["product", "feature"]))
            panels.append(macro_for_prod)
    macro_panel = pd.concat(panels, axis=1).sort_index(axis=1)

    # 合并 Group C + Group M_beta
    full = features_c.join(macro_panel).sort_index(axis=1)
    return full
