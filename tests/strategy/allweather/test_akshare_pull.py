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
    """12 个 ETF code 全部映射到有效 sina code (P1 后 7→12)."""
    assert len(ETF_CODE_MAP) == 12
    assert all(c.startswith("etf_") for c in ETF_CODE_MAP)
    assert all(c.startswith(("sh", "sz")) for c in ETF_CODE_MAP.values())


def test_fut_product_map_complete():
    """29 个期货 (23 商品 + 6 股指/国债) 全部映射到 sina 主连代码."""
    assert len(FUT_PRODUCT_MAP) == 29
    assert all(c.startswith("fut_") for c in FUT_PRODUCT_MAP)
    assert all(c.endswith("0") for c in FUT_PRODUCT_MAP.values())


def test_pull_etf_panel_returns_dataframe():
    """ETF 增量面板: 12 列对齐 panel, DatetimeIndex."""
    df = pull_etf_panel("2026-07-08", "2026-09-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 12
    assert all(c.startswith("etf_") for c in df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_pull_futures_panel_returns_dataframe():
    """期货增量面板: 29 列 (含 6 个新增股指/国债期货)."""
    df = pull_futures_panel("2026-07-08", "2026-09-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) >= 28  # 至少 28 个商品 (可能个别接口失败)
    assert all(c.startswith("fut_") for c in df.columns)


def test_v7_indices_p1():
    """P1: v7 指数拉取 (4 列: 沪深300/中证500/中证1000/中债10年)."""
    from QuantNodes.strategy.allweather.data_etl.akshare_pull import (
        pull_v7_indices, V7_INDEX_MAP,
    )
    assert len(V7_INDEX_MAP) == 4
    df = pull_v7_indices("2026-07-08", "2026-09-03")
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 4
    assert isinstance(df.index, pd.DatetimeIndex)


def test_p1_covers_unavailable_etfs():
    """P1: 之前不可用的 5 个 ETF 现在全部可拉到 2026-09-03."""
    df = pull_etf_panel("2026-07-08", "2026-09-03")
    for col in ["etf_512010", "etf_512690", "etf_515030", "etf_515050", "etf_512200"]:
        assert col in df.columns, f"缺 {col}"
        assert df[col].dropna().index.max() >= pd.Timestamp("2026-09-03"), \
            f"{col} 末 < 2026-09-03"


def test_p1_covers_index_futures():
    """P1: 6 个股指/国债期货现在全部可拉到 2026-09-03."""
    df = pull_futures_panel("2026-07-08", "2026-09-03")
    for col in ["fut_ic", "fut_if", "fut_ih", "fut_t", "fut_tf", "fut_ts"]:
        assert col in df.columns, f"缺 {col}"
        assert df[col].dropna().index.max() >= pd.Timestamp("2026-09-03"), \
            f"{col} 末 < 2026-09-03"


def test_window_filter():
    """窗口过滤: 拉取结果只在 [start_date, end_date] 范围内."""
    df = pull_etf_panel("2026-08-01", "2026-08-31")
    assert df.index.min() >= pd.Timestamp("2026-08-01")
    assert df.index.max() <= pd.Timestamp("2026-08-31")
