# coding=utf-8
"""GARCH 与波动率偏度因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.volatility import (
    GARCHFactor,
    VolSkewnessFactor,
)


def _mk_uptrend(n=300, start=100.0, end=200.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.linspace(start, end, n)}, index=idx)


class TestGARCH:
    def test_garch_positive_vol(self):
        prices = _mk_uptrend(n=300)
        out = GARCHFactor(window=150).compute(prices)
        # 训练后应有非 NaN 条件波动率
        valid = out["A"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()  # 波动率恒正

    def test_garch_fewer_rows_than_window(self):
        prices = _mk_uptrend(n=50)
        out = GARCHFactor(window=252).compute(prices)
        assert out["A"].isna().all()  # 样本不足 → 全 NaN

    def test_garch_shapes(self):
        prices = _mk_uptrend(n=300)
        out = GARCHFactor(window=150).compute(prices)
        assert out.shape == prices.shape
        assert out.index.equals(prices.index)


class TestVolSkewness:
    def test_constant_price_no_signal(self):
        # 常数价格 → 无偏度 → 0
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        prices = pd.DataFrame({"A": 100.0}, index=idx)
        sig = VolSkewnessFactor(60, 0.3).compute(prices)
        assert (sig["A"] == 0).all()

    def test_signal_values(self):
        prices = _mk_uptrend()
        sig = VolSkewnessFactor(60, 0.3).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_extreme_shock_positive_skew(self):
        # 大部分平稳 + 尾部大波动 → 正偏
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        rng = np.random.RandomState(42)
        rets = np.append(rng.normal(0, 0.002, 90), [0.05, -0.01, 0.03])
        rets = np.append(rets, rng.normal(0, 0.002, 7))
        prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)
        sig = VolSkewnessFactor(60, 0.3).compute(prices)
        # 应至少出现一次信号 (冲击造成偏度)
        assert (sig["A"] != 0).any()
