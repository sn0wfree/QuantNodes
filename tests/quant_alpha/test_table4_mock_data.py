# coding=utf-8
"""
test_table4_mock_data.py - MockDataLoader 测试
"""

from __future__ import annotations

import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation import MockDataLoader


class TestMockDataLoader:
    def test_default_construction(self):
        loader = MockDataLoader()
        assert loader.n_stocks == 500
        assert loader.n_days == 500
        assert loader.seed == 42

    def test_load_returns_dataframe(self):
        loader = MockDataLoader(n_stocks=10, n_days=20)
        df = loader.load()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 200

    def test_required_columns(self):
        loader = MockDataLoader(n_stocks=10, n_days=20)
        df = loader.load()
        required = {
            "date", "code", "open", "high", "low", "close",
            "vol", "amount", "industry",
        }
        assert required.issubset(set(df.columns))

    def test_industry_assignment(self):
        loader = MockDataLoader(n_stocks=20, n_days=10)
        df = loader.load()
        n_industries = df["industry"].n_unique()
        assert n_industries >= 2
        assert n_industries <= 10

    def test_seed_reproducibility(self):
        loader1 = MockDataLoader(n_stocks=10, n_days=20, seed=123)
        loader2 = MockDataLoader(n_stocks=10, n_days=20, seed=123)
        df1 = loader1.load()
        df2 = loader2.load()
        assert df1["close"].to_list() == df2["close"].to_list()

    def test_different_seed_different_data(self):
        loader1 = MockDataLoader(n_stocks=10, n_days=20, seed=1)
        loader2 = MockDataLoader(n_stocks=10, n_days=20, seed=2)
        df1 = loader1.load()
        df2 = loader2.load()
        assert df1["close"].to_list() != df2["close"].to_list()

    def test_load_summary(self):
        loader = MockDataLoader(n_stocks=20, n_days=30)
        summary = loader.load_summary()
        assert summary["n_stocks"] == 20
        assert summary["n_days"] == 30
        assert summary["n_rows"] == 600
        assert len(summary["industries"]) >= 2
        assert summary["close_mean"] > 0

    def test_forward_return_calculated(self):
        loader = MockDataLoader(n_stocks=10, n_days=20)
        df = loader.load()
        assert "forward_return_1d" in df.columns
        # 最后一个交易日 per stock 的 forward_return 应该为 null
        last_day_nulls = (
            df.group_by("code").agg(pl.col("forward_return_1d").tail(1).null_count())
        )
        assert last_day_nulls["forward_return_1d"].sum() == 10

    def test_prices_positive(self):
        loader = MockDataLoader(n_stocks=10, n_days=20)
        df = loader.load()
        assert df["close"].min() > 0
        assert df["high"].min() > 0
        assert df["low"].min() > 0