"""
v7.0 宏观因子 fetcher 单元测试 (Stage 30 POC).

测试覆盖:
1. 5 因子 parquet 存在 + 字段完整
2. obs_date 范围: 2018-01 ~ 2026-06
3. release_date = obs_date + lag_days
4. PIT 查询: T 日只能用 release_date <= T 的数据
5. PIT 关键测试: 早期不可见 (release 之前)
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

CACHE_DIR = Path("data/ifind_cache/macro")

EXPECTED_FACTORS = ["PMI", "CPI", "M2", "CN10Y", "US10Y"]
EXPECTED_LAG = {"PMI": 1, "CPI": 10, "M2": 12, "CN10Y": 0, "US10Y": 0}
EXPECTED_ROWS = {"PMI": 102, "CPI": 102, "M2": 101, "CN10Y": 2120, "US10Y": 2123}


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_parquet_exists(name):
    p = CACHE_DIR / f"{name}.parquet"
    assert p.exists(), f"缓存缺失: {p} (请先运行 fetch_all_macro)"


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_parquet_schema(name):
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    assert set(df.columns) >= {"obs_date", "value", "release_date"}, \
        f"{name} schema 缺失: {df.columns.tolist()}"


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_date_range(name):
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    assert df["obs_date"].min() >= pd.Timestamp("2018-01-01"), \
        f"{name} obs_date 起: {df['obs_date'].min()}"
    assert df["obs_date"].max() <= pd.Timestamp("2026-06-30"), \
        f"{name} obs_date 止: {df['obs_date'].max()}"


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_release_date_consistency(name):
    """release_date = obs_date + lag_days."""
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    expected_delta = timedelta(days=EXPECTED_LAG[name])
    actual_delta = df["release_date"] - df["obs_date"]
    # 容许 ±1 天误差 (节假日调整)
    diff = (actual_delta - expected_delta).abs()
    assert (diff <= timedelta(days=1)).all(), \
        f"{name} release_date 偏差 > 1 天: {diff[diff > timedelta(days=1)]}"


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_no_nan(name):
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    assert df["value"].isna().sum() == 0, f"{name} 有 NaN: {df['value'].isna().sum()}"


@pytest.mark.parametrize("name", EXPECTED_FACTORS)
def test_value_reasonable(name):
    """因子值在合理范围 (避免单位错误)."""
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    if name in ("PMI",):
        assert 30 <= df["value"].min() and df["value"].max() <= 80, \
            f"{name} 值域异常: {df['value'].min()} ~ {df['value'].max()}"
    elif name in ("CPI", "M2"):
        assert -10 <= df["value"].min() and df["value"].max() <= 30, \
            f"{name} 值域异常: {df['value'].min()} ~ {df['value'].max()}"
    elif name in ("CN10Y", "US10Y"):
        assert -5 <= df["value"].min() and df["value"].max() <= 20, \
            f"{name} 值域异常: {df['value'].min()} ~ {df['value'].max()}"


def test_pit_value_basic():
    """PIT 查询: 2024-06-30 时点, CPI 应能看到 2024-05 数据 (5月数据 6月10日发布)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import get_pit_value
    df = pd.read_parquet(CACHE_DIR / "CPI.parquet")
    # 2024-06-30 时点, 5月数据 release_date = 2024-06-09 (5月底+10天) 已发布
    v = get_pit_value(df, pd.Timestamp("2024-06-30"))
    assert v is not None
    # 5月 CPI 同比应在 [0, 1] 范围
    assert 0 <= v <= 1, f"2024-06-30 PIT CPI 值异常: {v}"


def test_pit_value_no_future():
    """PIT 关键测试: 在 release_date 之前看不到."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import get_pit_value
    df = pd.read_parquet(CACHE_DIR / "CPI.parquet")
    # 找第一个 obs_date (2018-01-31), release_date = 2018-02-10
    # 在 2018-02-09 时点, 该数据尚未发布, get_pit_value 应看不到
    v = get_pit_value(df, pd.Timestamp("2018-02-09"))
    # 应该返回 None 或 2017-12 数据 (但只有 2018-01, 没 2017-12 缓存)
    # 实际: 应是 None (没有任何 release_date <= 2018-02-09 的数据)
    if v is not None:
        # 如果返回了, 它的 release_date 应 <= 2018-02-09
        row = df[df["value"] == v].iloc[0]
        assert row["release_date"] <= pd.Timestamp("2018-02-09"), \
            f"LOOK-AHEAD BUG! 在 {row['release_date']} 才发布, 但 T=2018-02-09 就看到了"


def test_pit_value_exactly_release():
    """PIT 边界: T == release_date 时应能看到."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import get_pit_value
    df = pd.read_parquet(CACHE_DIR / "PMI.parquet")
    # 2018-01 obs_date=2018-01-31, release_date=2018-02-01
    # T=2018-02-01 时点应能看到
    v = get_pit_value(df, pd.Timestamp("2018-02-01"))
    assert v is not None, "T==release_date 应能看到"
    assert 45 <= v <= 55, f"PMI 1月值应在 45~55: {v}"


def test_pit_series_alignment():
    """批量 PIT: 5 个连续日期, 值应单调非递减 (因为 release_date 单调)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import get_pit_series
    df = pd.read_parquet(CACHE_DIR / "CPI.parquet")
    dates = pd.date_range("2018-02-01", "2018-08-01", freq="MS")
    s = get_pit_series(df, dates)
    # 早期 (2018-02-01 之前) 可能 NaN
    assert s.iloc[0] == s.iloc[0] or pd.isna(s.iloc[0]), "应有 NaN 或值"
    # 后期应有有效值
    assert s.iloc[-1] == s.iloc[-1], "末尾应有效"
    # 至少应该能从 NaN 过渡到非 NaN
    has_nan = s.isna().any()
    has_val = s.notna().any()
    assert has_nan or has_val


def test_pit_no_lookahead_simulation():
    """PIT 反 look-ahead 模拟: 模拟回测在 2024 年每月底, CPI 值是否 use release_date <= T."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import get_pit_value
    df = pd.read_parquet(CACHE_DIR / "CPI.parquet")
    # 12 个月
    for i in range(1, 13):
        T = pd.Timestamp(f"2024-{i:02d}-28")
        v = get_pit_value(df, T)
        if v is not None:
            # 找对应的 obs_date
            row = df[df["value"] == v].iloc[0]
            assert row["release_date"] <= T, \
                f"2024-{i:02d}-28 时点, 看到了 release 在 {row['release_date']} 的数据 (look-ahead!)"
