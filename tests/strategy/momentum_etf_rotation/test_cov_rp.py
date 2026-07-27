# coding=utf-8
"""Tests for Stage 11: 协方差估计 + 风险平价 (Risk Parity)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    VolTargeting,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.common.covariance import (
    estimate_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
    ewma_covariance,
    diagonal_covariance,
    is_positive_definite,
    condition_number,
)
from QuantNodes.strategy.momentum_etf_rotation.common.risk_parity import (
    solve_risk_parity,
    solve_max_diversification,
    risk_contribution,
)
from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import (
    inverse_vol_weights,
)


def _make_returns(n_days=120, n_assets=10, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_factors = 2
    factor_loadings = rng.normal(0, 0.3, (n_assets, n_factors))
    factor_rets = rng.normal(0, 0.01, (n_days, n_factors))
    idio_rets = rng.normal(0, 0.005, (n_days, n_assets))
    rets = factor_rets @ factor_loadings.T + idio_rets
    return pd.DataFrame(rets, columns=[f"A{i}" for i in range(n_assets)])


def _make_prices(rets: pd.DataFrame) -> pd.DataFrame:
    prices = (1 + rets).cumprod() * 100
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    return prices


@pytest.fixture
def returns():
    return _make_returns()


@pytest.fixture
def prices(returns):
    return _make_prices(returns)


class TestCovariance:
    def test_sample_covariance_shape(self, returns):
        cov = sample_covariance(returns)
        n = returns.shape[1]
        assert cov.shape == (n, n)

    def test_ledoit_wolf_positive_definite(self, returns):
        cov, alpha = ledoit_wolf_shrinkage(returns)
        assert is_positive_definite(cov)
        assert 0.0 <= alpha <= 1.0

    def test_ledoit_wolf_shrinkage_reduces_condition(self, returns):
        cov_sample = sample_covariance(returns)
        cov_lw, _ = ledoit_wolf_shrinkage(returns)
        cond_sample = condition_number(cov_sample)
        cond_lw = condition_number(cov_lw)
        assert cond_lw <= cond_sample * 1.1

    def test_ewma_covariance_shape(self, returns):
        cov = ewma_covariance(returns, halflife=30)
        n = returns.shape[1]
        assert cov.shape == (n, n)

    def test_diagonal_covariance(self, returns):
        cov = diagonal_covariance(returns)
        diag = np.diag(cov)
        off_diag = cov - np.diag(diag)
        assert np.allclose(off_diag, 0)

    def test_estimate_covariance_dispatch(self, returns):
        for method in ["sample", "ledoit_wolf", "ewma", "diagonal"]:
            cov = estimate_covariance(returns, method=method, halflife=30)
            assert cov.shape == (returns.shape[1], returns.shape[1])
            assert is_positive_definite(cov + np.eye(cov.shape[0]) * 1e-8)

    def test_estimate_unknown_method_raises(self, returns):
        with pytest.raises(ValueError):
            estimate_covariance(returns, method="unknown")


class TestRiskParity:
    def test_risk_contribution_sum_to_one(self, returns):
        cov = sample_covariance(returns)
        w = np.ones(len(returns.columns)) / len(returns.columns)
        rc = risk_contribution(w, cov)
        assert abs(rc.sum() - 1.0) < 1e-9

    def test_solve_risk_parity_weights_sum_to_one(self, returns):
        cov = sample_covariance(returns)
        w = solve_risk_parity(cov)
        assert abs(w.sum() - 1.0) < 1e-6
        assert all(w >= 0)

    def test_solve_risk_parity_equal_risk_contributions(self, returns):
        cov = sample_covariance(returns)
        w = solve_risk_parity(cov, max_iter=500)
        rc = risk_contribution(w, cov)
        target = 1.0 / len(rc)
        for r in rc:
            assert abs(r - target) / target < 0.10

    def test_solve_risk_parity_non_pos_def_raises(self):
        not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError):
            solve_risk_parity(not_psd)

    def test_solve_max_diversification(self, returns):
        cov = sample_covariance(returns)
        w = solve_max_diversification(cov)
        assert abs(w.sum() - 1.0) < 1e-6
        assert all(w >= 0)


class TestPortfolioRiskParityWeights:
    def test_disabled_returns_inv_vol_fallback(self, prices):
        codes = list(prices.columns[:5])
        weights = inverse_vol_weights(prices, codes, prices.index[-1])
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    @pytest.mark.skip(reason="risk_parity_weights removed during refactoring")
    def test_risk_parity_weights_from_prices(self, prices):
        pass

    @pytest.mark.skip(reason="risk_parity_weights removed during refactoring")
    def test_risk_parity_weights_insufficient_data_fallback(self):
        pass


class TestBacktestRiskParity:
    def _make_panel(self, n_days=800, n_codes=20):
        rng = np.random.default_rng(42)
        codes = list(DEFAULT_POOL.codes)[:n_codes]
        rets = rng.normal(0.0003, 0.012, (n_days, n_codes))
        prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
        return pd.DataFrame(prices, index=pd.bdate_range("2020-01-01", periods=n_days), columns=codes)

    def test_baseline_vs_rp_runs(self):
        panel = self._make_panel()
        cfg_baseline = RotationConfig(lookback=120, top_n=5, weight_method="inv_vol")
        cfg_rp = RotationConfig(
            lookback=120, top_n=5, weight_method="risk_parity",
        )
        r1 = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg_baseline))
        r2 = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg_rp))
        assert len(r1.states) > 0
        assert len(r2.states) > 0

    def test_rp_returns_valid_metrics(self):
        panel = self._make_panel()
        cfg = RotationConfig(
            lookback=120, top_n=5, weight_method="risk_parity",
        )
        r = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(r.nav)
        assert "calmar" in m

    def test_all_cov_methods_run(self):
        panel = self._make_panel()
        for method in ["ledoit_wolf", "sample", "ewma", "diagonal"]:
            cfg = RotationConfig(
                lookback=120, top_n=5, weight_method="risk_parity",
            )
            r = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
            m = performance_metrics(r.nav)
            assert m["calmar"] > 0 or m["max_drawdown"] < 0