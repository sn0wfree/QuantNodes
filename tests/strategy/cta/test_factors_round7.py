# coding=utf-8
"""第七轮因子测试 (资金流/量能指标)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.money_flow import (
    CMFFactor,
    ForceIndexFactor,
    MFIFactor,
    PVTFactor,
)
from QuantNodes.strategy.cta.factors.volume_indicators import (
    EaseOfMovementFactor,
    KlingerOscillatorFactor,
    VolumeMACrossFactor,
    VolumeRatioFactor,
)


def _mk_prices(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _mk_ohlcv(n=300):
    prices = _mk_prices(n)
    idx = prices.index
    rng = np.random.RandomState(7)
    return pd.DataFrame({
        "open": prices["A"] * 0.995, "high": prices["A"] * 1.01,
        "low": prices["A"] * 0.99, "close": prices["A"],
        "volume": rng.randint(1000, 5000, n),
    }, index=idx)


def _mk_volumes(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(3)
    return pd.DataFrame({
        "A": rng.randint(1000, 5000, n),
        "B": rng.randint(1000, 5000, n),
    }, index=idx)


class TestMFI:
    def test_output(self):
        ohlcv = _mk_ohlcv()
        sig = MFIFactor(14, 20, 80).compute(ohlcv)
        assert set(sig["mfi"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_high_mfi(self):
        # 持续上涨 + 放量 → MFI 高位
        n = 150
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ohlcv = pd.DataFrame({
            "open": np.linspace(100, 200, n),
            "high": np.linspace(101, 201, n),
            "low": np.linspace(99, 199, n),
            "close": np.linspace(100, 200, n),
            "volume": np.linspace(1000, 5000, n),
        }, index=idx)
        sig = MFIFactor(14, 20, 80).compute(ohlcv)
        # 超买信号应出现
        assert (sig["mfi"] == -1.0).any()


class TestCMF:
    def test_output(self):
        ohlcv = _mk_ohlcv()
        sig = CMFFactor(20, 0.1).compute(ohlcv)
        assert set(sig["cmf"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_close_at_high_positive(self):
        # close 恒 = high → CLV 正 → CMF 正 → 多
        n = 150
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ohlcv = pd.DataFrame({
            "high": np.full(n, 101.0), "low": np.full(n, 99.0),
            "close": np.full(n, 101.0), "volume": np.full(n, 1000.0),
        }, index=idx)
        sig = CMFFactor(20, 0.1).compute(ohlcv)
        assert (sig["cmf"] > 0).any()


class TestPVT:
    def test_output(self):
        prices = _mk_prices()
        vols = _mk_volumes()
        sig = PVTFactor(20, 1.0).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestForceIndex:
    def test_output(self):
        prices = _mk_prices()
        vols = _mk_volumes()
        sig = ForceIndexFactor(13, 1.0).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVolumeRatio:
    def test_expansion(self):
        # 放量段 → 量比 > 1.5
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        vol = np.concatenate([np.full(100, 1000.0), np.full(20, 3000.0)])
        volumes = pd.DataFrame({"A": vol}, index=idx)
        sig = VolumeRatioFactor(5, 1.5, 0.5).compute(volumes)
        assert (sig["A"].iloc[-20:] > 0).any()

    def test_output(self):
        volumes = _mk_volumes()
        sig = VolumeRatioFactor(5, 1.5, 0.5).compute(volumes)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVolumeMACross:
    def test_expansion_cross(self):
        # 前平后放量 → 短量均线 > 长量均线
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        vol = np.concatenate([np.full(110, 1000.0), np.full(10, 5000.0)])
        volumes = pd.DataFrame({"A": vol}, index=idx)
        sig = VolumeMACrossFactor(5, 20).compute(volumes)
        # 最后一天: short(5d)=5000 > long(20d)=(10×1000+10×5000)/20=3000
        assert sig["A"].iloc[-1] == 1.0

class TestEaseOfMovement:
    def test_output(self):
        ohlcv = _mk_ohlcv()
        sig = EaseOfMovementFactor(14, 1.0).compute(ohlcv)
        assert set(sig["emv"].unique()).issubset({-1.0, 0.0, 1.0})


class TestKlinger:
    def test_output(self):
        ohlcv = _mk_ohlcv()
        sig = KlingerOscillatorFactor(34, 55).compute(ohlcv)
        assert set(sig["klinger"].unique()).issubset({-1.0, 0.0, 1.0})
