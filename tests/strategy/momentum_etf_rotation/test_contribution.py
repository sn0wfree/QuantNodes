# coding=utf-8
"""Tests for contribution.py (5 维度归因)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.contribution import (
    DEFAULT_PERIODS,
    category_contribution,
    etf_contribution,
    marginal_contribution,
    period_contribution,
    reconstruct_daily_weights,
    risk_contribution,
)
from QuantNodes.strategy.momentum_etf_rotation.universe import (
    Category, ETFMeta, ETFPool,
)


def _make_pool(n_codes: int = 5) -> ETFPool:
    """合成小池."""
    members = tuple(
        ETFMeta(
            code=f"E{i:03d}", name=f"E{i:03d}",
            category=Category.A_BROAD if i % 2 == 0 else Category.HK,
            index_code=f"I{i % 3}", liquidity_rank=1,
        )
        for i in range(n_codes)
    )
    return ETFPool(members=members)


def _make_panel(n_days: int = 252, n_codes: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"E{i:03d}" for i in range(n_codes)]
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, (n_days, n_codes))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=codes)


def _make_weights(n_days: int = 252, n_codes: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    w = rng.uniform(0, 0.3, (n_days, n_codes))
    w = w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=idx, columns=[f"E{i:03d}" for i in range(n_codes)])


class TestReconstructDailyWeights:
    def test_returns_dataframe(self) -> None:
        """应返回 DataFrame 索引为 trading_dates."""
        from dataclasses import dataclass

        @dataclass
        class FakeState:
            weights: dict

        states = [FakeState({"E000": 0.5, "E001": 0.5}), FakeState({"E000": 0.3, "E002": 0.7})]
        rebal_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")]
        trading_dates = pd.bdate_range("2020-01-01", "2020-12-31")
        df = reconstruct_daily_weights(states, rebal_dates, trading_dates)
        assert isinstance(df, pd.DataFrame)
        # 3 个唯一 codes: E000, E001, E002
        assert df.shape == (len(trading_dates), 3)
        # 在第一个 rebal 日, 应等于 state[0].weights
        assert df.iloc[0]["E000"] == 0.5
        assert df.iloc[0]["E001"] == 0.5
        # 在第二个 rebal 日后, 应等于 state[1].weights (ffill)
        second_rebal_idx = trading_dates.get_loc(rebal_dates[1])
        assert df.iloc[second_rebal_idx]["E000"] == 0.3
        assert df.iloc[second_rebal_idx]["E002"] == 0.7

    def test_empty_states(self) -> None:
        """空 states 应返回空 DataFrame."""
        trading_dates = pd.bdate_range("2020-01-01", periods=10)
        df = reconstruct_daily_weights([], [], trading_dates)
        assert df.empty


class TestEtfContribution:
    def test_returns_expected_columns(self) -> None:
        """应包含 5 个期望列."""
        nav_df = _make_panel()
        weights_df = _make_weights()
        pool = _make_pool()
        df = etf_contribution(nav_df, weights_df, pool)
        assert "code" in df.columns
        assert "frequency" in df.columns
        assert "avg_weight" in df.columns
        assert "total_return" in df.columns
        assert "return_contrib" in df.columns

    def test_sorted_by_contribution(self) -> None:
        """应按 return_contrib 降序排列."""
        nav_df = _make_panel()
        weights_df = _make_weights()
        pool = _make_pool()
        df = etf_contribution(nav_df, weights_df, pool)
        if len(df) > 1:
            contribs = df["return_contrib"].values
            assert all(contribs[i] >= contribs[i + 1] for i in range(len(contribs) - 1))


class TestCategoryContribution:
    def test_returns_expected_columns(self) -> None:
        """应包含 5 个期望列."""
        nav_df = _make_panel()
        weights_df = _make_weights()
        pool = _make_pool()
        df = category_contribution(weights_df, nav_df, pool)
        assert "category" in df.columns
        assert "avg_weight" in df.columns
        assert "return_contrib" in df.columns
        assert "frequency" in df.columns
        assert "n_codes" in df.columns


class TestRiskContribution:
    def test_returns_dataframe(self) -> None:
        """应返回 DataFrame 含 vol_contrib 和 var_contrib."""
        weights_df = _make_weights()
        cov = np.eye(5) * 0.0004  # 5x5 identity cov
        df = risk_contribution(weights_df, cov)
        assert isinstance(df, pd.DataFrame)
        assert "code" in df.columns
        assert "vol_contrib" in df.columns
        assert "var_contrib" in df.columns


class TestMarginalContribution:
    def test_returns_dataframe(self) -> None:
        """应返回 DataFrame 含 marginal_sharpe."""
        weights_df = _make_weights()
        returns_df = _make_panel()
        cov = np.eye(5) * 0.0004
        df = marginal_contribution(weights_df, returns_df, cov)
        assert isinstance(df, pd.DataFrame)
        assert "code" in df.columns
        assert "marginal_sharpe" in df.columns


class TestPeriodContribution:
    def test_returns_dataframe(self) -> None:
        """应返回 DataFrame 含 period 和 calmar."""
        weights_df = _make_weights()
        nav_df = _make_panel()
        df = period_contribution(weights_df, nav_df, periods=DEFAULT_PERIODS[:2])
        assert isinstance(df, pd.DataFrame)
        assert "period" in df.columns
        assert "calmar" in df.columns
        assert "ann_return" in df.columns

    def test_uses_default_periods(self) -> None:
        """不指定 periods 应使用 DEFAULT_PERIODS."""
        weights_df = _make_weights()
        nav_df = _make_panel(n_days=1500)  # 长数据覆盖多个周期
        df = period_contribution(weights_df, nav_df)
        assert len(df) == len(DEFAULT_PERIODS)