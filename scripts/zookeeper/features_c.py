# coding: utf-8
"""Group C (商品特定) 变量构造 — P2 待实装.

论文 37 变量分三族: 库存/对冲/流动性. 本项目可构造 ~28 (缺 hp12).

参考: docs/cta/zookeeper_paper_notes.md §9.1
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd


LOG = logging.getLogger(__name__)


def build_group_c(conn: duckdb.DuckDBPyConnection,
                  products: list[str],
                  start: str,
                  end: str) -> pd.DataFrame:
    """构造 28 个 Group C 变量 (月度截面面板).

    Returns:
        DataFrame: index=[trade_date], columns=MultiIndex (product, feature)
        月频 (MonthEnd), PIT 对齐 (滞后 1 期防月末信息).
    """
    # TODO P2: 沿论文 §3.1 公式 + 本项目数据映射实现
    # 库存族: basis / rbasis (spot_basis_daily) / mom12 (smoothed_{p} 12月收益)
    #        / basis_term_slope (term_structure_daily rank1-rank2/days)
    #        / inv_yoy (inventory_daily YoY) / inv_to_consume (库存消费比)
    # 对冲族: dOI (OI 月变化) / dOI_skew (OI 变动偏度). hp12 缺 (⚠️ cfric 覆盖不全)
    # 流动性族: bmom (volume × sign(ret)) / iliq (abs(ret)/volume) / atv (vol 滚动均值)
    #          / turnover (volume / OI)
    raise NotImplementedError("P2 待实装")
