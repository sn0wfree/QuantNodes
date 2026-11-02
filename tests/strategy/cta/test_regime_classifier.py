# coding=utf-8
"""Regime 状态识别子模块测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.regime import (
    ARGarchEstimator,
    STATE_RANGE,
    STATE_TREND_DOWN,
    STATE_TREND_UP,
    STATE_UNKNOWN,
    classify_regime,
    classify_regime_from_estimator,
    compute_tsi_panel,
    kaufman_adaptive_window,
    tsi_bootstrap_ci,
    tsi_point,
)


def _mk_strong_trend(n=400, drift=0.003, vol=0.005, seed=42):
    """强趋势: 持续正漂移 + 低噪声."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rets = rng.normal(drift, vol, n)
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)


def _mk_pure_noise(n=400, vol=0.015, seed=42):
    """纯随机游走: 期望 0 漂移."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rets = rng.normal(0.0, vol, n)
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)


def _mk_ar1_prices(n=400, phi=0.5, drift=0.001, vol=0.01, seed=42):
    """AR(1) 驱动的对数价格."""
    rng = np.random.RandomState(seed)
    eps = rng.normal(0, vol, n)
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = drift + phi * ret[t - 1] + eps[t]
    log_p = np.cumsum(ret) + np.log(100)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.exp(log_p)}, index=idx)


class TestKaufmanAdaptiveWindow:
    def test_window_in_range(self):
        prices = _mk_ar1_prices()
        windows = kaufman_adaptive_window(prices["A"])
        valid = windows.dropna()
        assert valid.min() >= 21
        assert valid.max() <= 250

    def test_trend_short_window(self):
        # 强趋势 → ER 高 → 窗口短
        prices = _mk_strong_trend()
        windows = kaufman_adaptive_window(prices["A"])
        # 后期应明显小于 max
        later_mean = windows.iloc[200:].dropna().mean()
        assert later_mean < 250

    def test_noise_long_window(self):
        # 纯噪声 → ER 低 → 窗口长
        prices = _mk_pure_noise()
        windows = kaufman_adaptive_window(prices["A"])
        later_mean = windows.iloc[200:].dropna().mean()
        assert later_mean > 100


class TestARGarchEstimator:
    def test_basic_output(self):
        prices = _mk_strong_trend()
        params = ARGarchEstimator(base_window=60, use_adaptive=False).estimate(prices)
        assert {"mu", "rho", "sigma"}.issubset(params.keys())
        for v in params.values():
            assert v.shape == prices.shape

    def test_adaptive_includes_window(self):
        prices = _mk_strong_trend()
        params = ARGarchEstimator(use_adaptive=True).estimate(prices)
        assert "window_used" in params

    def test_uptrend_positive_mu(self):
        prices = _mk_strong_trend(drift=0.003)
        params = ARGarchEstimator(use_adaptive=False).estimate(prices)
        assert params["mu"]["A"].dropna().mean() > 0


class TestTSIBootstrap:
    def test_tsi_point_positive_for_uptrend(self):
        prices = _mk_strong_trend()
        est = ARGarchEstimator(use_adaptive=False)
        params = est.estimate(prices)
        tsi = tsi_point(params["mu"]["A"], params["sigma"]["A"], window=60)
        assert tsi.dropna().mean() > 0

    def test_tsi_bootstrap_ci_columns(self):
        prices = _mk_strong_trend()
        params = ARGarchEstimator(use_adaptive=False).estimate(prices)
        out = tsi_bootstrap_ci(params["mu"]["A"], params["sigma"]["A"])
        assert list(out.columns) == ["tsi", "tsi_lo", "tsi_hi"]
        # lo ≤ tsi ≤ hi
        valid = out.dropna()
        assert (valid["tsi_lo"] <= valid["tsi"]).all()
        assert (valid["tsi"] <= valid["tsi_hi"]).all()

    def test_compute_panel_shape(self):
        prices = _mk_strong_trend()
        params = ARGarchEstimator(use_adaptive=False).estimate(prices)
        panel = compute_tsi_panel(params)
        assert panel.shape == (len(prices), 3)  # 单标的, 3 列
        assert ("A", "tsi") in panel.columns


class TestClassifyRegime:
    def _build_inputs(self, prices):
        """辅助: 端到端构造 classifier 输入."""
        params = ARGarchEstimator(use_adaptive=False).estimate(prices)
        panel = compute_tsi_panel(params)
        tsi_lo = panel.xs("tsi_lo", axis=1, level=1)
        tsi_hi = panel.xs("tsi_hi", axis=1, level=1)
        return params["mu"], params["rho"], params["sigma"], tsi_lo, tsi_hi

    def test_strong_trend_detected(self):
        prices = _mk_strong_trend(drift=0.005)
        mu, rho, sigma, lo, hi = self._build_inputs(prices)
        states = classify_regime(mu, rho, sigma, lo, hi, confirm_window=5)
        # 强趋势 → 至少应有部分 trend_up 状态
        assert (states["A"] == STATE_TREND_UP).sum() > 0

    def test_pure_noise_no_trend(self):
        prices = _mk_pure_noise()
        mu, rho, sigma, lo, hi = self._build_inputs(prices)
        states = classify_regime(mu, rho, sigma, lo, hi, confirm_window=10)
        # 纯噪声 → trend 信号不应占主导 (随机游走下 false positive 不可避免)
        trend_ratio = states["A"].isin(
            [STATE_TREND_UP, STATE_TREND_DOWN],
        ).mean()
        assert trend_ratio < 0.25  # < 25% 时间

    def test_state_values_in_set(self):
        prices = _mk_strong_trend()
        mu, rho, sigma, lo, hi = self._build_inputs(prices)
        states = classify_regime(mu, rho, sigma, lo, hi)
        allowed = {STATE_UNKNOWN, STATE_RANGE, STATE_TREND_UP, STATE_TREND_DOWN}
        assert set(states["A"].unique()).issubset(allowed)

    def test_initial_unknown(self):
        # 样本不足初期 → unknown
        prices = _mk_strong_trend(n=80)
        mu, rho, sigma, lo, hi = self._build_inputs(prices)
        states = classify_regime(mu, rho, sigma, lo, hi, confirm_window=10)
        # 至少前 60 行应为 unknown (base_window 60)
        assert (states["A"].iloc[:59] == STATE_UNKNOWN).all()

    def test_end_to_end(self):
        prices = _mk_strong_trend()
        params = ARGarchEstimator(use_adaptive=False).estimate(prices)
        panel = compute_tsi_panel(params)
        states = classify_regime_from_estimator(params, panel, confirm_window=5)
        assert (states["A"] == STATE_TREND_UP).sum() > 0
