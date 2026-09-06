# coding: utf-8
"""Group M (宏观) 变量构造 — P2 实装.

论文 40 个宏观. 本项目基本面库可构造 10 个:
  fed_rate / term_spread / fed_balance / m2_yoy / us_cpi_yoy
  / dxy / vix / tips10y / epu / wti_brent_spread

S 族跳过 (用户决策): Baker-Wurgler / FinBERT WSJ 缺数据.

PIT: 整体滞后 1 期 (防御性, 沿用 §26 审计).
鲁棒性: Spanning Test 追加 pit_lag=2 对照.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import loader
from .config import GROUP_M_FEATURES

LOG = logging.getLogger(__name__)

# value_col → 计算方式 (直接取/衍生)
M_DEFS = {
    "fed_rate": "fed_rate_daily",
    "term_spread": "us_10y_2y_spread",
    "fed_balance": "fed_total_assets",
    "m2_yoy": "m2_money_supply",       # 取原值, Spanning Test 中可做 pct_change(12) 衍生
    "us_cpi_yoy": "us_cpi_yoy",
    "dxy": "dxy",
    "vix": "vix",
    "tips10y": "tips10y",
    "epu": "epu",
    # wti_brent_spread 需衍生 (wti - brent)
}


def build_group_m(start: str, end: str, pit_lag: int = 1) -> pd.DataFrame:
    """构造 10 个 Group M 宏观变量 → 月频面板.

    Args:
        start/end: yyyy-mm-dd.
        pit_lag: 滞后月数 (默认 1, 防御性). Spanning Test 追加 pit_lag=2 鲁棒性对照.

    Returns:
        DataFrame: index=MonthEnd, columns=feature. 缺失留 NaN.
    """
    LOG.info(f"P2 Group M: {len(GROUP_M_FEATURES)} 变量, pit_lag={pit_lag}")
    series = {}
    for feat in GROUP_M_FEATURES:
        if feat == "wti_brent_spread":
            wti = loader.load_fundamental_monthly("wti", start, end, pit_lag=pit_lag)
            brent = loader.load_fundamental_monthly("brent", start, end, pit_lag=pit_lag)
            if wti.empty or brent.empty:
                series[feat] = pd.Series(dtype=float)
            else:
                union_idx = wti.index.union(brent.index)
                series[feat] = (wti.reindex(union_idx) - brent.reindex(union_idx))
        else:
            vc = M_DEFS[feat]
            series[feat] = loader.load_fundamental_monthly(vc, start, end, pit_lag=pit_lag)
    panel = pd.DataFrame(series)
    panel = panel.sort_index()
    return panel
