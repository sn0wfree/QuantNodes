# coding=utf-8
"""AR(1) 含截距回归因子测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.statistical_ext import AR1Factor


def _mk_uptrend(n=300, start=100.0, end=200.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.linspace(start, end, n)}, index=idx)


def _mk_random_walk(n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)


def _mk_ar1_prices(n=500, phi=0.3, innovation_scale=0.01, seed=42):
    """构造 AR(1) 收益驱动的对数价格序列: ret[t] = φ·ret[t-1] + ε.

    AR1Factor 内部对 prices 做 pct_change 估计 AR(1),
    对数价格的小收益序列 → pct_change ≈ ret, 故 rho 估计 ≈ φ.
    """
    rng = np.random.RandomState(seed)
    eps = rng.normal(0, innovation_scale, n)
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = phi * ret[t - 1] + eps[t]
    log_p = np.cumsum(ret) + np.log(100)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.exp(log_p)}, index=idx)


class TestAR1:
    def test_raw_value_shape(self):
        prices = _mk_uptrend()
        params = AR1Factor(window=60).raw_value(prices)
        assert set(params.keys()) == {"mu", "rho", "sigma"}
        for k, df in params.items():
            assert df.shape == prices.shape
            assert df.index.equals(prices.index)

    def test_initial_nan(self):
        # 前 window-1 行应全为 NaN (样本不足)
        prices = _mk_uptrend(n=120)
        params = AR1Factor(window=60).raw_value(prices)
        for k, df in params.items():
            assert df.iloc[:59].isna().all().all()
            # 之后开始有值
            assert df.iloc[59:].notna().any().any()

    def test_uptrend_positive_mu(self):
        # 持续上行 → 估计 μ > 0
        params = AR1Factor(window=60).raw_value(_mk_uptrend())
        valid_mu = params["mu"]["A"].dropna()
        assert valid_mu.mean() > 0

    def test_positive_rho_momentum(self):
        # 模拟 AR(1) φ=0.3 (动量) → 估计 ρ 应接近 0.3
        prices = _mk_ar1_prices(phi=0.3)
        params = AR1Factor(window=60).raw_value(prices)
        valid_rho = params["rho"]["A"].dropna()
        assert valid_rho.mean() > 0.1  # 应明显为正

    def test_negative_rho_reversal(self):
        # 模拟 AR(1) φ=-0.3 (反转) → 估计 ρ 应明显为负
        prices = _mk_ar1_prices(phi=-0.3)
        params = AR1Factor(window=60).raw_value(prices)
        valid_rho = params["rho"]["A"].dropna()
        assert valid_rho.mean() < -0.1

    def test_random_walk_rho_near_zero(self):
        # 随机游走 → ρ 应在零附近
        params = AR1Factor(window=60).raw_value(_mk_random_walk(n=500))
        valid_rho = params["rho"]["A"].dropna()
        assert abs(valid_rho.mean()) < 0.15

    def test_sigma_positive(self):
        # σ 应始终 ≥ 0
        params = AR1Factor(window=60).raw_value(_mk_uptrend())
        valid_sigma = params["sigma"]["A"].dropna()
        assert (valid_sigma >= 0).all()

    def test_compute_shape(self):
        # compute() 返回单列信号
        prices = _mk_uptrend()
        sig = AR1Factor(window=60).compute(prices)
        assert sig.shape == prices.shape
        assert sig.index.equals(prices.index)

    def test_compute_signal_values(self):
        sig = AR1Factor(window=60).compute(_mk_uptrend())
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_compute_momentum_signal(self):
        # φ=0.3 动量序列 → 后期应有大量 +1 信号
        prices = _mk_ar1_prices(phi=0.3)
        sig = AR1Factor(window=60).compute(prices)
        # 后期 (估计稳态后) 多头占比应 > 反向
        later = sig.iloc[200:]
        assert (later["A"] == 1.0).sum() > (later["A"] == -1.0).sum()

    def test_short_data_returns_nan(self):
        # 样本不足 → 全 NaN
        prices = pd.DataFrame({"A": [100.0, 101.0]}, index=pd.date_range("2025-01-01", periods=2))
        params = AR1Factor(window=60).raw_value(prices)
        for k, df in params.items():
            assert df.isna().all().all()

    def test_name(self):
        assert AR1Factor.name == "ar1"
