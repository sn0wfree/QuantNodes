# coding=utf-8
"""第五轮因子测试 (价格位置/趋势质量/量能退潮)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.price_location import (
    OpenLocationFactor,
    PriceLocationFactor,
    VWAPDeviationFactor,
)
from QuantNodes.strategy.cta.factors.trend_quality import (
    MomentumSlopeFactor,
    TrendQualityFactor,
)
from QuantNodes.strategy.cta.factors.volume_decay import (
    VolatilityConeFactor,
    VolumeDecayFactor,
)
from QuantNodes.strategy.cta.factors.momentum_xs import AmihudXSFactor


def _mk_prices(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)  # 单调趋势
    c = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))  # 震荡
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def _mk_ohlcv(n=260):
    prices = _mk_prices(n)
    idx = prices.index
    return pd.DataFrame({
        "open": prices["A"] * 0.995, "high": prices["A"] * 1.01,
        "low": prices["A"] * 0.99, "close": prices["A"],
        "volume": np.random.RandomState(7).randint(1000, 5000, n),
    }, index=idx)


def _mk_volumes(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # 先放量后缩量
    vol = np.concatenate([np.linspace(1000, 5000, n // 2),
                          np.linspace(5000, 1000, n - n // 2)])
    return pd.DataFrame({"A": vol}, index=idx)


class TestPriceLocation:
    def test_uptrend_high_location(self):
        prices = _mk_prices()
        sig = PriceLocationFactor(20, 1.0).compute(prices)
        # B 单调上行 → 位置 z 高 → 多头
        assert (sig["B"] > 0).any()

    def test_output_values(self):
        prices = _mk_prices()
        sig = PriceLocationFactor(20, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVWAPDeviation:
    def test_close_above_vwap(self):
        ohlcv = _mk_ohlcv()
        # 构造: close 持续高于 typical → 正偏离
        ohlcv["close"] = ohlcv["high"]  # close = high > typical
        sig = VWAPDeviationFactor(5, 0.001).compute(ohlcv)
        assert (sig["vwap_dev"] > 0).any()

    def test_output_values(self):
        ohlcv = _mk_ohlcv()
        sig = VWAPDeviationFactor(5, 0.005).compute(ohlcv)
        assert set(sig["vwap_dev"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_missing_col(self):
        import pytest
        with pytest.raises(ValueError):
            VWAPDeviationFactor().compute(_mk_prices())


class TestOpenLocation:
    def test_output_values(self):
        ohlc = _mk_ohlcv()[["open", "high", "low", "close"]]
        sig = OpenLocationFactor(5, 1.0).compute(ohlc)
        assert set(sig["open_loc"].unique()).issubset({-1.0, 0.0, 1.0})


class TestTrendQuality:
    def test_pure_trend_high_r2(self):
        prices = _mk_prices()
        sig = TrendQualityFactor(30, 0.6).compute(prices)
        # B 纯线性 → 有信号; C 震荡 → 少信号
        assert (sig["B"] != 0).sum() > (sig["C"] != 0).sum()

    def test_output_values(self):
        prices = _mk_prices()
        sig = TrendQualityFactor(30, 0.6).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestMomentumSlope:
    def test_output_values(self):
        prices = _mk_prices()
        sig = MomentumSlopeFactor(20, 10, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVolumeDecay:
    def test_decay_signal(self):
        volumes = _mk_volumes()
        sig = VolumeDecayFactor(60, 10, 0.5).compute(volumes)
        # 后半段缩量 → 退潮信号
        assert (sig["A"].iloc[-20:] > 0).any()


class TestVolatilityCone:
    def test_output_values(self):
        prices = _mk_prices()
        sig = VolatilityConeFactor(10, 120, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestAmihudXS:
    def test_low_illiq_long(self):
        prices = _mk_prices()
        # A 高量, B 中量 (线性低波 → illiq 最低), C 低量 (流动性最差)
        n = len(prices)
        idx = prices.index
        volumes = pd.DataFrame({
            "A": np.full(n, 10000.0),
            "B": np.full(n, 5000.0),
            "C": np.full(n, 500.0),
        }, index=idx)
        sig = AmihudXSFactor(20, 0.33).compute(prices, volumes)
        last = sig.iloc[-1]
        # B 线性低波 → illiq 最低 → 多; C 低量高波 → illiq 最高 → 空
        assert last["B"] == 1.0
        assert last["C"] == -1.0
