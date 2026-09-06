# coding: utf-8
"""数据加载 + PIT 对齐 — P2 待实装.

PIT 纪律 (沿用 tuning_log §26):
  - 月度频率, 月末取截面, **滞后 1 期** 防月末提前发布信息
  - 宏观变量按"修正后最终值"取, Spanning Test 中加鲁棒性对照
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd

from .config import DUCKDB_PATH

LOG = logging.getLogger(__name__)


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """主行情库连接."""
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def load_smoothed_close(products: list[str],
                        start: str, end: str) -> pd.DataFrame:
    """加载 60 产品 smoothed_{p}_daily close 序列 (用于月动量/回测).

    Returns:
        DataFrame: index=[trade_date], columns=product. 日频.
    """
    # TODO P2: 拼接各产品 smoothed_{p}.close, pivoting by product
    raise NotImplementedError("P2 待实装")


def monthly_returns(daily_close: pd.DataFrame) -> pd.DataFrame:
    """日频 close → 月频收益 (MonthEnd 截面)."""
    # TODO P2: resample('M').last().pct_change()
    raise NotImplementedError("P2 待实装")


def load_fundamental(coll: str, start: str, end: str) -> pd.Series:
    """从基本面库取单变量月度序列."""
    # TODO P2: ATTACH fundamentals.duckdb, query value_col=coll, 月末取截面, PIT 滞后 1 期
    raise NotImplementedError("P2 待实装")


def load_cost_model() -> dict:
    """加载成本校准 (沿用 §22)."""
    # TODO P4: json.load(COST_MODEL)
    raise NotImplementedError("P4 待实装")
