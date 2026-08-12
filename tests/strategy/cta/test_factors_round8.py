# coding=utf-8
"""第八轮因子测试 (振荡/趋势进阶/均值回归)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.oscillators import (
    CCIFactor,
    MACDFactor,
    StochasticFactor,
    WilliamsRFactor,
)
from QuantNodes.strategy.cta.factors.trend_advanced import (
    AroonFactor,
    ChandelierExitFactor,
    LinearSlopeFactor,
    TRIXFactor,
    VortexFactor,
)
from QuantNodes.strategy.cta.factors.mean_reversion import (
    HalfLifeFactor,
    SqueezeFactor,
)


def _mk_prices(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _mk_ohlc(n=300):
    prices = _mk_prices(n)
    return pd.DataFrame({
        "open": prices["A"] * 0.995, "high": prices["A"] * 1.01,
        "low": prices["A"] * 0.99, "close": prices["A"],
    }, index=prices.index)


class TestMACD:
    def test_cross_signals(self):
        prices = _mk_prices()
        sig = MACDFactor(12, 26, 9).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_cross_up(self):
        # 先横后涨 → 金叉
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        s = np.concatenate([np.full(100, 100.0), np.linspace(100, 150, 100)])
        prices = pd.DataFrame({"A": s}, index=idx)
        sig = MACDFactor(12, 26, 9).compute(prices)
        assert (sig["A"] > 0).any()


class TestStochastic:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = StochasticFactor(14, 3).compute(ohlc)
        assert set(sig["stochastic"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_overbought(self):
        # 强上行 → %K 高位 → 超买 → 交叉回落信号
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ohlc = pd.DataFrame({
            "high": np.linspace(101, 201, n), "low": np.linspace(99, 199, n),
            "close": np.linspace(100, 200, n),
        }, index=idx)
        sig = StochasticFactor(14, 3).compute(ohlc)
        assert set(sig["stochastic"].unique()).issubset({-1.0, 0.0, 1.0})


class TestCCI:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = CCIFactor(20, 100, -100).compute(ohlc)
        assert set(sig["cci"].unique()).issubset({-1.0, 0.0, 1.0})


class TestWilliamsR:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = WilliamsRFactor(14, -80, -20).compute(ohlc)
        assert set(sig["williams_r"].unique()).issubset({-1.0, 0.0, 1.0})


class TestAroon:
    def test_output(self):
        prices = _mk_prices()
        sig = AroonFactor(25).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_up_cross(self):
        prices = _mk_prices()
        sig = AroonFactor(25).compute(prices)
        # B 单调上行 → Aroon_Up 恒高 → 有上升信号
        assert (sig["B"] > 0).any()


class TestVortex:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = VortexFactor(14).compute(ohlc)
        assert set(sig["vortex"].unique()).issubset({-1.0, 0.0, 1.0})


class TestLinearSlope:
    def test_output(self):
        prices = _mk_prices()
        sig = LinearSlopeFactor(20, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_positive(self):
        prices = _mk_prices()
        sig = LinearSlopeFactor(20, 1.0).compute(prices)
        assert (sig["B"] > 0).any()


class TestTRIX:
    def test_output(self):
        prices = _mk_prices()
        sig = TRIXFactor(15).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestChandelier:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = ChandelierExitFactor(22, 14, 3.0).compute(ohlc)
        assert set(sig["chandelier"].unique()).issubset({-1.0, 0.0, 1.0})


class TestHalfLife:
    def test_output(self):
        prices = _mk_prices()
        sig = HalfLifeFactor(60, 10, 30).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_mean_reverting_short_hl(self):
        # 强均值回归序列 → 短半衰期 → 反转信号
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.RandomState(5)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.9 * y[t - 1] + rng.normal(0, 1)  # 强回归 (b=-0.1)
        prices = pd.DataFrame({"A": 100 + y}, index=idx)
        sig = HalfLifeFactor(60, 10, 30).compute(prices)
        assert (sig["A"] != 0).any()


class TestSqueeze:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = SqueezeFactor(20, 2.0, 1.5).compute(ohlc)
        assert set(sig["squeeze"].unique()).issubset({-1.0, 0.0, 1.0})
