# coding: utf-8
"""Group M (宏观) 变量构造 — P2 待实装.

论文 40 个. 本项目基本面库 fundamentals_daily 可构造 ~10.
S 族跳过 (用户决策).

参考: docs/cta/zookeeper_paper_notes.md §9.2
"""
from __future__ import annotations

import logging

import pandas as pd


LOG = logging.getLogger(__name__)


def build_group_m(start: str, end: str) -> pd.DataFrame:
    """构造 10 个 Group M 宏观变量 (月度面板, PIT 对齐).

    Returns:
        DataFrame: index=[trade_date (MonthEnd)], columns=feature
        来源: fundamentals_daily (data/cmf/fundamental.duckdb)
        PIT: 宏观变量发布延迟 1-2 月 (沿用 §26 审计纪律).
    """
    # TODO P2: 从 fundamentals_daily 取 value_col 月末截面
    # fed_rate / term_spread (10Y-2Y) / fed_balance / m2_yoy / us_cpi_yoy
    # dxy / vix / tips10y / epu / wti_brent_spread
    # PIT: 延迟 1 期使用 (防止月末提前发布信息)
    # 鲁棒性测试: Spanning Test 中追加"加 1-2 月延迟"对照
    raise NotImplementedError("P2 待实装")
