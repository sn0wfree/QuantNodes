"""v7.0 5 Macro Dynamic 方案单元测试 (Stage 30.5).

[测试覆盖]
- Top-K Dynamic (4 测试)
- Black-Litterman (6 测试)
- Macro Beta Regression (4 测试)
- State Conditional Momentum (3 测试)
- State Conditional Inverse Vol (3 测试)
- 共享 (2 测试: PIT simulation + 5 策略 metrics)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    compute_state_conditional_means,
    compute_dynamic_topk_weights,
    run_topk_v7_backtest,
    compute_expanding_cov,
    compute_state_view_q,
    compute_view_uncertainty_omega,
    compute_bl_posterior,
    compute_bl_weights,
    run_bl_v7_backtest,
    compute_etf_macro_betas,
    predict_etf_returns,
    run_beta_v7_backtest,
    compute_etf_momentum,
    run_momentum_v7_backtest,
    compute_etf_vol,
    compute_inverse_vol_weights,
    run_iv_v7_backtest,
    build_regime_timeline,
)


ETFS = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']


def _load_panel():
    nav_main = pd.read_parquet('data/real/etf_nav_2018-01-01_2026-06-30.parquet')
    sb = pd.read_parquet('data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet')
    panel = pd.DataFrame()
    for c in ETFS:
        if c in nav_main.columns:
            s = nav_main[c].dropna()
        elif c in sb.columns:
            s = sb[c].dropna()
        else:
            continue
        panel[c] = s
    return panel.dropna(how='all').ffill().dropna()


def _load_timeline():
    tl = build_regime_timeline()
    tl['date'] = pd.to_datetime(tl['date'])
    return tl.set_index('date')


@pytest.fixture(scope="module")
def panel():
    return _load_panel()


@pytest.fixture(scope="module")
def tl_df():
    return _load_timeline()


# ====== Top-K Dynamic Tests ======

def test_state_conditional_means_shape(panel, tl_df):
    means = compute_state_conditional_means(panel, tl_df, panel.index[-1])
    assert 'state' in means.columns
    assert len(means) > 0
    assert set(means['state'].unique()).issubset(
        {'recovery', 'overheat', 'neutral', 'stagflation', 'recession'}
    )


def test_state_conditional_means_pit(panel, tl_df):
    cutoff = panel.index[300]
    means = compute_state_conditional_means(panel, tl_df, cutoff)
    assert means.index.max() <= cutoff
    for d in means.index:
        idx = panel.index.get_loc(d)
        assert idx + 21 < len(panel.loc[:cutoff])


def test_topk_weights_sum_one(panel, tl_df):
    means = compute_state_conditional_means(panel, tl_df, panel.index[-1])
    w = compute_dynamic_topk_weights(means, 'recovery', k=5)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())
    nonzero = sum(1 for v in w.values() if v > 0)
    assert nonzero <= 5


def test_topk_cold_start(panel, tl_df):
    means = pd.DataFrame()
    w = compute_dynamic_topk_weights(means, 'recovery', k=5, etf_universe=ETFS)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_topk_runs(panel, tl_df):
    nav_df, weights_df, m = run_topk_v7_backtest(panel, tl_df, k=5)
    assert 'nav_cum' in nav_df.columns
    assert nav_df['nav_cum'].iloc[-1] > 1.0
    assert -0.5 < m['dd'] < 0
    assert 0.0 < m['calmar'] < 5.0


# ====== Black-Litterman Tests ======

def test_expanding_cov_shape(panel, tl_df):
    sigma = compute_expanding_cov(panel, panel.index[-1], window=252)
    n = len(ETFS)
    assert sigma.shape == (n, n)
    assert np.allclose(sigma, sigma.T)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() > -1e-6


def test_state_view_q_pit(panel, tl_df):
    cutoff = panel.index[-1]
    means = compute_state_conditional_means(panel, tl_df, cutoff)
    q = compute_state_view_q(means, 'recovery')
    assert q is not None
    assert len(q) == len(ETFS)
    assert not np.isnan(q).any()


def test_bl_posterior_no_views():
    n = 5
    pi = np.full(n, 0.07)
    sigma = np.eye(n) * 0.04
    p = np.eye(n)
    q = np.zeros(n)
    omega = np.eye(n) * 100
    post = compute_bl_posterior(pi, sigma, p, q, omega, tau=0.05)
    np.testing.assert_allclose(post, pi, atol=1e-4)


def test_bl_posterior_with_views():
    n = 5
    pi = np.full(n, 0.07)
    sigma = np.eye(n) * 0.04
    p = np.eye(n)
    q = np.array([0.20, 0.05, 0.07, 0.07, 0.07])
    omega = np.eye(n) * 0.01
    post = compute_bl_posterior(pi, sigma, p, q, omega, tau=0.05)
    assert post[0] > pi[0]
    assert post[1] < pi[1]


def test_bl_weights_long_only():
    n = 5
    post = np.array([0.20, -0.05, 0.10, 0.07, 0.15])
    sigma = np.eye(n) * 0.04
    w = compute_bl_weights(post, sigma, max_weight=0.40)
    assert all(v >= 0 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.40


def test_view_uncertainty_omega(panel, tl_df):
    cutoff = panel.index[800]
    means = compute_state_conditional_means(panel, tl_df, cutoff)
    sigma = compute_expanding_cov(panel, cutoff, window=252)
    omega = compute_view_uncertainty_omega(means, 'recovery', sigma, tau=0.05)
    assert omega.shape == (len(ETFS), len(ETFS))
    assert np.allclose(omega, np.diag(np.diag(omega)))


def test_bl_runs(panel, tl_df):
    nav_df, weights_df, m = run_bl_v7_backtest(panel, tl_df, tau=0.05, max_weight=0.30)
    assert 'nav_cum' in nav_df.columns
    assert nav_df['nav_cum'].iloc[-1] > 1.0
    assert -0.5 < m['dd'] < 0
    assert 0.0 < m['calmar'] < 5.0


# ====== Macro Beta Regression Tests ======

def test_etf_macro_betas_shape(panel, tl_df):
    betas = compute_etf_macro_betas(panel, tl_df, panel.index[800], lookback=252)
    assert betas.shape == (len(ETFS), 6)
    assert list(betas.columns) == ['const', 'PMI', 'CPI', 'M2', 'CN10Y', 'US10Y']


def test_predict_etf_returns():
    betas = pd.DataFrame(
        np.array([[0.0, 0.5, 0.3, 0.2, 0.1, 0.0]] * 7),
        index=ETFS,
        columns=['const', 'PMI', 'CPI', 'M2', 'CN10Y', 'US10Y']
    )
    macro = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=['PMI', 'CPI', 'M2', 'CN10Y', 'US10Y'])
    pred = predict_etf_returns(betas, macro)
    assert len(pred) == 7
    assert pred.iloc[0] > 0


def test_beta_cold_start(panel, tl_df):
    betas = compute_etf_macro_betas(panel, tl_df, panel.index[10], lookback=252, min_samples=60)
    assert betas is None


def test_beta_runs(panel, tl_df):
    nav_df, weights_df, m = run_beta_v7_backtest(panel, tl_df, lookback=252, k=5)
    assert 'nav_cum' in nav_df.columns
    assert nav_df['nav_cum'].iloc[-1] > 1.0
    assert -0.5 < m['dd'] < 0
    assert 0.0 < m['calmar'] < 5.0


# ====== State Conditional Momentum Tests ======

def test_etf_momentum(panel):
    mom = compute_etf_momentum(panel, panel.index[-1], lookback=63)
    assert len(mom) == len(ETFS)
    assert not mom.isna().all()


def test_momentum_pit(panel, tl_df):
    cutoff = panel.index[500]
    nav_df, weights_df, m = run_momentum_v7_backtest(panel.loc[:cutoff], tl_df.loc[:cutoff], lookback=63, k=5)
    assert m['ann'] >= -0.5


def test_momentum_runs(panel, tl_df):
    nav_df, weights_df, m = run_momentum_v7_backtest(panel, tl_df, lookback=63, k=5)
    assert 'nav_cum' in nav_df.columns
    assert nav_df['nav_cum'].iloc[-1] > 1.0
    assert -0.5 < m['dd'] < 0
    assert 0.0 < m['calmar'] < 5.0


# ====== State Conditional Inverse Vol Tests ======

def test_etf_vol_shape(panel):
    vol = compute_etf_vol(panel, panel.index[-1], lookback=252)
    assert len(vol) == len(ETFS)
    assert (vol > 0).all()


def test_iv_weights_long_only(panel):
    vol = compute_etf_vol(panel, panel.index[-1], lookback=252)
    w = compute_inverse_vol_weights(vol, max_weight=0.30)
    assert all(v >= 0 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.30


def test_iv_runs(panel, tl_df):
    nav_df, weights_df, m = run_iv_v7_backtest(panel, tl_df, lookback=252, max_weight=0.30)
    assert 'nav_cum' in nav_df.columns
    assert nav_df['nav_cum'].iloc[-1] > 1.0
    assert -0.5 < m['dd'] < 0
    assert 0.0 < m['calmar'] < 5.0


# ====== 共享测试 ======

def test_pit_no_lookahead_simulation(panel, tl_df):
    cutoff = panel.index[500]
    panel_sub = panel.loc[:cutoff]
    tl_sub = tl_df.loc[:cutoff]
    nav_df, _, m = run_topk_v7_backtest(panel_sub, tl_sub, k=5)
    assert nav_df.index.max() <= cutoff
    for d in nav_df.index[1:]:
        assert d in panel_sub.index


def test_5_strategies_same_metrics_shape(panel, tl_df):
    runners = [
        run_topk_v7_backtest(panel, tl_df, k=5),
        run_bl_v7_backtest(panel, tl_df, tau=0.05),
        run_beta_v7_backtest(panel, tl_df, lookback=252, k=5),
        run_momentum_v7_backtest(panel, tl_df, lookback=63, k=5),
        run_iv_v7_backtest(panel, tl_df, lookback=252),
    ]
    keys = ['ann', 'vol', 'sharpe', 'dd', 'calmar']
    for nav_df, weights_df, m in runners:
        for k in keys:
            assert k in m
            assert isinstance(m[k], float)
            assert -2.0 < m[k] < 5.0
