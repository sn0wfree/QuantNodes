# coding=utf-8
"""tests/strategy/momentum_etf_rotation/v9/test_factor_galaxy.py

v9 银河因子配置单元测试.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v9.factor_galaxy import (
    entropy_weight,
    composite_score,
    rolling_factor_beta,
    variance_decomposition,
    risk_budget_weights,
    galaxy_factor_allocation,
    compute_factor_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_allocator import (
    map_to_categories,
    run_factor_allocator,
    CATEGORY_MAPPING,
)


def make_macro_df(n_periods=300, n_factors=5, seed=42):
    """构造测试用宏观数据."""
    np.random.seed(seed)
    idx = pd.date_range('2020-01-01', periods=n_periods, freq='W')
    cols = [f'factor_{i}' for i in range(n_factors)]
    data = np.random.randn(n_periods, n_factors).cumsum(axis=0)
    return pd.DataFrame(data, index=idx, columns=cols)


def make_returns_df(n_periods=300, n_assets=5, seed=42):
    """构造测试用资产收益."""
    np.random.seed(seed)
    idx = pd.date_range('2020-01-01', periods=n_periods, freq='W')
    cols = [f'asset_{i}' for i in range(n_assets)]
    data = np.random.randn(n_periods, n_assets) * 0.02
    return pd.DataFrame(data, index=idx, columns=cols)


class TestEntropyWeight:
    def test_basic(self):
        df = make_macro_df(n_periods=200)
        weights = entropy_weight(df, window=104)

        assert isinstance(weights, dict)
        assert len(weights) == df.shape[1]
        assert all(0 <= w <= 1 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_short_data_returns_equal(self):
        df = make_macro_df(n_periods=50)
        weights = entropy_weight(df, window=104)
        expected = 1.0 / df.shape[1]
        assert all(abs(w - expected) < 1e-10 for w in weights.values())


class TestCompositeScore:
    def test_basic(self):
        df = make_macro_df()
        weights = {c: 0.2 for c in df.columns}
        score = composite_score(df, weights)
        assert len(score) == len(df)
        assert isinstance(score, pd.Series)


class TestRollingFactorBeta:
    def test_shape(self):
        returns = make_returns_df()
        factor_score = make_macro_df(n_periods=300, n_factors=1).iloc[:, 0]
        beta = rolling_factor_beta(returns, factor_score, window=52)

        assert beta.shape == returns.shape
        assert beta.iloc[:52].isna().all().all() or (beta.iloc[:52] == 0).all().all()


class TestVarianceDecomposition:
    def test_range(self):
        returns = make_returns_df()
        factor_score = make_macro_df(n_periods=300, n_factors=1).iloc[:, 0]
        beta = rolling_factor_beta(returns, factor_score, window=52)
        contrib = variance_decomposition(returns, factor_score, beta, window=52)

        assert contrib.shape[0] == returns.shape[0]
        valid = contrib.iloc[52:]
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()


class TestRiskBudgetWeights:
    def test_basic(self):
        returns = make_returns_df()
        factor_score = make_macro_df(n_periods=300, n_factors=1).iloc[:, 0]
        beta = rolling_factor_beta(returns, factor_score, window=52)
        weights = risk_budget_weights(returns, factor_score, beta, window=52)

        assert weights.shape == returns.shape
        sums = weights.sum(axis=1).dropna()
        assert (sums.abs() < 1e-6).sum() / len(sums) < 0.5

    def test_cap(self):
        returns = make_returns_df()
        factor_score = make_macro_df(n_periods=300, n_factors=1).iloc[:, 0]
        beta = rolling_factor_beta(returns, factor_score, window=52)
        weights = risk_budget_weights(returns, factor_score, beta, window=52, cap=0.50, floor=0.0)

        assert weights.shape == returns.shape
        assert (weights.max().max() <= 0.50 + 1e-6)


class TestGalaxyFactorAllocation:
    def test_integration(self):
        macro = make_macro_df(n_periods=300, n_factors=5)
        returns = make_returns_df(n_periods=300, n_assets=5)

        weights, factor_score, betas = galaxy_factor_allocation(
            returns_df=returns,
            macro_indicators=macro,
            lookback_score=104,
            lookback_beta=52,
        )

        assert weights.shape[1] == returns.shape[1]
        assert len(factor_score) == returns.shape[0]

    def test_insufficient_data_raises(self):
        macro = make_macro_df(n_periods=50, n_factors=5)
        returns = make_returns_df(n_periods=50, n_assets=5)

        with pytest.raises(ValueError):
            galaxy_factor_allocation(
                returns_df=returns,
                macro_indicators=macro,
                lookback_score=104,
                lookback_beta=52,
            )


class TestMapToCategories:
    def test_mapping(self):
        macro = pd.DataFrame({
            '宏观增长因子': np.random.randn(100),
            '宏观通胀因子_生活端': np.random.randn(100),
            '宏观汇率因子': np.random.randn(100),
            'unknown_factor': np.random.randn(100),
        }, index=pd.date_range('2020-01-01', periods=100, freq='W'))

        cat = map_to_categories(macro)
        assert 'unknown_factor' not in cat.columns
        assert '消费/内需' in cat.columns
        assert '出口/外部' in cat.columns


class TestRunFactorAllocator:
    def test_integration(self):
        macro = make_macro_df(n_periods=300, n_factors=5)
        for cat_name, factors in CATEGORY_MAPPING.items():
            for factor in factors:
                if factor not in macro.columns:
                    macro[factor] = np.random.randn(300).cumsum()

        returns = make_returns_df(n_periods=300, n_assets=5)

        weights, factor_score, betas, cat_df = run_factor_allocator(
            returns_df=returns,
            macro_df=macro,
            lookback_score=104,
            lookback_beta=52,
        )

        assert cat_df.shape[1] == macro.shape[1]
        assert weights.shape[1] == 5
        valid_sums = weights.sum(axis=1).dropna()
        assert valid_sums.abs().max() < 1e-6 or (valid_sums < 1.1).all()


class TestComputeFactorMetrics:
    def test_basic(self):
        n = 252
        idx = pd.date_range('2020-01-01', periods=n, freq='D')
        returns = pd.DataFrame(
            np.random.randn(n, 3) * 0.01,
            index=idx, columns=['A', 'B', 'C'],
        )
        weights = pd.DataFrame(1.0/3, index=idx[:52], columns=returns.columns)

        metrics = compute_factor_metrics(weights, returns)
        assert 'Sharpe' in metrics
        assert 'Calmar' in metrics
        assert 'MaxDD' in metrics
        assert 'AnnRet' in metrics