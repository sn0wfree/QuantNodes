# coding: utf-8
"""60 月滚动 OLS Beta (CS 维度对齐) — P3 待实装.

论文 §3.1 关键设计: 宏观/情绪不随品种变, 用 60 月滚动 OLS Beta
构造商品级 β_g,i / β_s,i, 实现 81→多维商品面板.

⚠️ P1 脚手架: 接口稳定, 实现留 P3.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import BETA_ROLL_WINDOW

LOG = logging.getLogger(__name__)


def rolling_beta(returns: pd.DataFrame, macro: pd.DataFrame,
                 window: int = BETA_ROLL_WINDOW) -> pd.DataFrame:
    """60 月滚动 OLS Beta.

    Args:
        returns: 商品收益 DataFrame, columns=product, index=trade_date (MonthEnd).
        macro: 宏观变量 DataFrame, columns=feature, index=trade_date (MonthEnd).
        window: 滚动窗口 (默认 60 月).

    Returns:
        Beta DataFrame: index=trade_date, columns=MultiIndex (product, macro_feature).
    """
    # TODO P3: 对每 (product, macro_feature) 对, 滚动 OLS
    # beta = cov(r_i, m_g, window) / var(m_g, window)
    # 月末对齐, NaN 在前 60 月填充
    raise NotImplementedError("P3 待实装")


def build_commodity_panel(features_c: pd.DataFrame, features_m_beta: pd.DataFrame,
                          features_s_beta: pd.DataFrame | None = None) -> pd.DataFrame:
    """合成商品面板: Group C 原始 + (Group M/S × 商品级 Beta).

    Returns:
        DataFrame: columns=MultiIndex (product, feature), index=trade_date.
        S 族本项目跳过, features_s_beta=None.
    """
    # TODO P3: concat columns (product × feature)
    raise NotImplementedError("P3 待实装")
