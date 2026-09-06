# coding: utf-8
"""数据加载 + PIT 对齐 — P2 实装.

PIT 纪律 (沿用 tuning_log §26 审计):
  - 月度频率, 月末取截面, **滞后 1 期** 防月末提前发布信息
  - 宏观 (fundamentals_daily) 滞后 1 期整体使用 (防御性)
  - 库存 (inventory_daily) 按数据方记账日口径直接用 (已审计)
  - 行情类 (smoothed/spot_basis/term_structure) 月末 close/截面, 天然 PIT OK
  - Spanning Test 阶段追加"加 1-2 月延迟"鲁棒性对照 (知乎风险 §7.2)
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd

from .config import (DUCKDB_PATH, FUNDAMENTAL_DB)

LOG = logging.getLogger(__name__)


def get_market_conn(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """主行情库连接."""
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def get_fundamental_conn(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """基本面库连接."""
    return duckdb.connect(str(FUNDAMENTAL_DB), read_only=read_only)


def load_smoothed_close(products: list[str], start: str, end: str) -> pd.DataFrame:
    """加载 60 产品 smoothed_{p}.close 日频序列.

    Returns:
        DataFrame: index=trade_date (date), columns=product. 日频.
    """
    con = get_market_conn()
    try:
        frames = []
        for p in products:
            tbl = f"smoothed_{p.lower()}_daily"
            r = con.execute(
                f"SELECT trade_date, close FROM {tbl} "
                f"WHERE trade_date BETWEEN '{start}' AND '{end}' "
                f"ORDER BY trade_date"
            ).fetchall()
            if not r:
                continue
            df = pd.DataFrame(r, columns=["trade_date", p])
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            frames.append(df.set_index("trade_date"))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1).sort_index()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
        return out
    finally:
        con.close()


def month_end_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """日频 panel → 月末截面 (MonthEnd)."""
    if daily.empty:
        return daily
    idx = pd.DatetimeIndex(daily.index)
    me = idx.to_period("M").to_timestamp("M")
    return daily.groupby(me).last()


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """日频 close → 月频收益 (MonthEnd 截面)."""
    me = month_end_panel(daily)
    return me.pct_change()


def load_fundamental_monthly(coll: str, start: str, end: str,
                             pit_lag: int = 1) -> pd.Series:
    """从基本面库取单变量月度序列, PIT 滞后 pit_lag 期.

    Args:
        coll: fundamentals_daily.value_col (如 'fed_rate', 'vix').
        start/end: yyyy-mm-dd.
        pit_lag: 滞后月数 (默认 1). Spanning Test 中追加 pit_lag=2 鲁棒性对照.

    Returns:
        Series: index=MonthEnd date, values=value. 缺失留 NaN.
    """
    con = get_fundamental_conn()
    try:
        r = con.execute(
            f"SELECT trade_date, value FROM fundamentals_daily "
            f"WHERE value_col='{coll}' AND trade_date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY trade_date"
        ).fetchall()
        if not r:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r, columns=["trade_date", "value"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        s = df.set_index("trade_date")["value"].astype(float)
        s.index = pd.DatetimeIndex(s.index)
        me = s.groupby(s.index.to_period("M").to_timestamp("M")).last()
        if pit_lag > 0:
            me = me.shift(pit_lag)
        return me
    finally:
        con.close()


def load_inventory_monthly(value_col: str, start: str, end: str) -> pd.Series:
    """从基本面库 inventory_daily 取单变量月度序列 (按数据方记账日).

    PIT: 数据方记账日口径, 不额外滞后 (已审计 §26).
    Returns: Series: index=MonthEnd.
    """
    con = get_fundamental_conn()
    try:
        r = con.execute(
            f"SELECT trade_date, value FROM inventory_daily "
            f"WHERE value_col='{value_col}' AND trade_date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY trade_date"
        ).fetchall()
        if not r:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r, columns=["trade_date", "value"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        s = df.set_index("trade_date")["value"].astype(float)
        s.index = pd.DatetimeIndex(s.index)
        return s.groupby(s.index.to_period("M").to_timestamp("M")).last()
    finally:
        con.close()


def load_tfapi_monthly(index_name: str, start: str, end: str) -> pd.Series:
    """从 tfapi_indicators 取单变量月度序列."""
    con = get_fundamental_conn()
    try:
        r = con.execute(
            f"SELECT trade_date, value FROM tfapi_indicators "
            f"WHERE index_name='{index_name}' AND trade_date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY trade_date"
        ).fetchall()
        if not r:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r, columns=["trade_date", "value"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        s = df.set_index("trade_date")["value"].astype(float)
        s.index = pd.DatetimeIndex(s.index)
        return s.groupby(s.index.to_period("M").to_timestamp("M")).last()
    finally:
        con.close()


def load_cost_model() -> dict:
    """加载成本校准 (沿用 §22 cost_model.json)."""
    import json
    from .config import COST_MODEL
    if not COST_MODEL.exists():
        LOG.warning(f"cost_model.json 不存在 ({COST_MODEL}), 使用默认 15bps")
        return {"default_one_side_bps": 15, "products": {}}
    return json.loads(COST_MODEL.read_text())
