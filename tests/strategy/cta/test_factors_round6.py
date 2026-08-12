# coding=utf-8
"""第六轮因子测试 (K线形态/协整/市场结构)."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.factors.candlestick import (
    DojiFactor,
    EngulfingFactor,
    HammerFactor,
    LongBodyFactor,
)
from QuantNodes.strategy.cta.factors.cointegration import (
    PairRatioMeanReversionFactor,
    SpreadMeanReversionFactor,
)
from QuantNodes.strategy.cta.factors.market_structure import (
    NewHighLowFactor,
    VolumeReturnResidualFactor,
)
from QuantNodes.strategy.cta.factors.trend import ATRBreakoutFactor, BIASFactor
from QuantNodes.strategy.cta.factors.statistical import FractalDimensionFactor


def _mk_prices(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)
    c = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def _mk_ohlc(n=300):
    prices = _mk_prices(n)
    return pd.DataFrame({
        "open": prices["A"] * 0.995, "high": prices["A"] * 1.01,
        "low": prices["A"] * 0.99, "close": prices["A"],
    }, index=prices.index)


class TestLongBody:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = LongBodyFactor(20, 0.6).compute(ohlc)
        assert set(sig["long_body"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_big_bull_candle(self):
        # 构造大阳线: close 远高于 open
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ohlc = pd.DataFrame({
            "open": np.full(n, 100.0), "close": np.full(n, 101.0),
            "high": np.full(n, 101.5), "low": np.full(n, 99.5),
        }, index=idx)
        sig = LongBodyFactor(20, 0.6).compute(ohlc)
        assert (sig["long_body"] > 0).any()


class TestDoji:
    def test_doji_detected(self):
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ohlc = pd.DataFrame({
            "open": np.full(n, 100.0), "close": np.full(n, 100.01),
            "high": np.full(n, 102.0), "low": np.full(n, 98.0),
        }, index=idx)
        sig = DojiFactor(0.1).compute(ohlc)
        assert (sig["doji"] > 0).any()


class TestHammer:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = HammerFactor(2.0).compute(ohlc)
        assert set(sig["hammer"].unique()).issubset({-1.0, 0.0, 1.0})


class TestEngulfing:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = EngulfingFactor(0.3).compute(ohlc)
        assert set(sig["engulfing"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_bull_engulfing(self):
        # 昨阴今阳且今实体吞昨实体
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        close[50] = 98.0   # 昨日阴 (prev close < prev open)
        open_[51] = 97.0   # 今日低开
        close[51] = 101.0  # 今日大阳吞没
        high = np.full(n, 102.0)
        low = np.full(n, 96.0)
        ohlc = pd.DataFrame({"open": open_, "close": close,
                             "high": high, "low": low}, index=idx)
        sig = EngulfingFactor(0.2).compute(ohlc)
        assert (sig["engulfing"] > 0).any()


class TestSpreadMR:
    def test_high_z_short(self):
        # 价差恒高 → 均值回归 → 空
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        spread = pd.Series(np.full(n, 100.0), index=idx)
        spread.iloc[-1] = 110.0  # 突发偏离
        sig = SpreadMeanReversionFactor(60, 1.5).compute(spread)
        assert sig.iloc[-1] == -1.0

    def test_output(self):
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        spread = pd.Series(np.random.RandomState(1).normal(0, 1, n), index=idx)
        sig = SpreadMeanReversionFactor(60, 1.5).compute(spread)
        assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})


class TestPairRatioMR:
    def test_output(self):
        prices = _mk_prices()
        sig = PairRatioMeanReversionFactor("A", "B", 60, 1.5).compute(prices)
        assert sig.columns[0] == "A_B_mr"
        assert set(sig.iloc[:, 0].unique()).issubset({-1.0, 0.0, 1.0})

    def test_missing(self):
        prices = _mk_prices()
        with pytest.raises(KeyError):
            PairRatioMeanReversionFactor("A", "ZZ").compute(prices)


class TestNewHighLow:
    def test_uptrend_positive(self):
        prices = _mk_prices()
        sig = NewHighLowFactor(60, 1.0).compute(prices)
        # B 单调上行 → 持续创新高 → 多头
        assert (sig["B"] > 0).any()

    def test_output(self):
        prices = _mk_prices()
        sig = NewHighLowFactor(60, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVolumeReturnResidual:
    def test_output(self):
        prices = _mk_prices()
        vols = pd.DataFrame({c: np.random.RandomState(7).randint(1000, 5000, len(prices))
                             for c in prices.columns}, index=prices.index)
        sig = VolumeReturnResidualFactor(30, 1.0).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestATRBreakout:
    def test_output(self):
        ohlc = _mk_ohlc()
        sig = ATRBreakoutFactor(14, 2.0).compute(ohlc)
        assert set(sig["atr_breakout"].unique()).issubset({-1.0, 0.0, 1.0})


class TestBIAS:
    def test_extreme_up_short(self):
        # 价格远超均线 → 负乖离过大? 正乖离 → 空
        prices = _mk_prices()
        sig = BIASFactor(20, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_output(self):
        prices = _mk_prices()
        sig = BIASFactor(20, 1.0).compute(prices)
        assert set(sig["B"].unique()).issubset({-1.0, 0.0, 1.0})


class TestFractalDimension:
    def test_output(self):
        prices = _mk_prices()
        sig = FractalDimensionFactor(120, 20, 0.05).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})
