"""data_etl.akshare_pull 单元测试."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.allweather.data_etl.akshare_pull import (
    ETF_CODE_MAP,
    FUT_PRODUCT_MAP,
    pull_etf_panel,
    pull_futures_panel,
)


def test_etf_code_map_complete():
    """7 个 ETF code 全部映射到有效 sina code."""
    assert len(ETF_CODE_MAP) == 7
    assert all(c.startswith("etf_") for c in ETF_CODE_MAP)
    assert all(c.startswith(("sh", "sz")) for c in ETF_CODE_MAP.values())


def test_fut_product_map_complete():
    """23 个商品全部映射到有效 sina 主连代码."""
    assert len(FUT_PRODUCT_MAP) == 23
    assert all(c.startswith("fut_") for c in FUT_PRODUCT_MAP)
    assert all(c.endswith("0") for c in FUT_PRODUCT_MAP.values())


def test_pull_etf_panel_returns_dataframe():
    """ETF 增量面板: 列名对齐 panel (etf_xxxxxx), 索引为 date."""
    df = pull_etf_panel("2026-07-08", "2026-09-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 7
    assert all(c.startswith("etf_") for c in df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_pull_futures_panel_returns_dataframe():
    """期货增量面板: 23 列对齐 panel (fut_xx), close 单价."""
    df = pull_futures_panel("2026-07-08", "2026-09-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) >= 20  # 至少 20 个商品 (可能个别接口失败)
    assert all(c.startswith("fut_") for c in df.columns)


def test_window_filter():
    """窗口过滤: 拉取结果只在 [start_date, end_date] 范围内."""
    df = pull_etf_panel("2026-08-01", "2026-08-31")
    assert df.index.min() >= pd.Timestamp("2026-08-01")
    assert df.index.max() <= pd.Timestamp("2026-08-31")
