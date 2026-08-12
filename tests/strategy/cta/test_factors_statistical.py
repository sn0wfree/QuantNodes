# coding=utf-8
"""CTA 二轮因子 (统计/微观结构/复合) 测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.statistical import (
    AutocorrelationFactor,
    HurstExponentFactor,
    SkewnessFactor,
    VarianceRatioFactor,
    hurst_exponent,
    variance_ratio,
)
from QuantNodes.strategy.cta.factors.microstructure import (
    CloseLocationFactor,
    GapFactor,
    IntradayReturnFactor,
    OvernightIntradayDivergenceFactor,
    OvernightReturnFactor,
)
from QuantNodes.strategy.cta.factors.composite import (
    MomentumCrashFilterFactor,
    MultiHorizonMomentumFactor,
    SharpeMomentumFactor,
)


def _mk_uptrend(n=200, start=100.0, end=200.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.linspace(start, end, n)}, index=idx)


def _mk_ohlc(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    base = np.linspace(100, 200, n)
    return pd.DataFrame({
        "open": base - 1, "high": base + 2, "low": base - 2, "close": base,
    }, index=idx)


class TestHurst:
    def test_hurst_uptrend_persistent(self):
        # 单调趋势 → H > 0.5
        s = _mk_uptrend()["A"]
        h = hurst_exponent(s, max_lag=30)
        assert h > 0.5

    def test_hurst_random_walk(self):
        # 随机游走 → H ≈ 0.5
        rng = np.random.RandomState(42)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
        h = hurst_exponent(s, max_lag=30)
        assert 0.2 < h < 0.8

    def test_factor_output(self):
        sig = HurstExponentFactor(120, 20, 0.05).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVarianceRatio:
    def test_vr_uptrend_positive(self):
        s = _mk_uptrend()["A"]
        vr = variance_ratio(s, k=5)
        assert vr > 0.5

    def test_factor_output(self):
        sig = VarianceRatioFactor(120, 5, 0.1).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestAutocorrelation:
    def test_factor_output(self):
        sig = AutocorrelationFactor(60, 0.1).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestSkewness:
    def test_factor_output(self):
        sig = SkewnessFactor(60, 0.3).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestMicrostructure:
    def test_overnight_uptrend(self):
        # open > prev close → 正隔夜
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        base = np.linspace(100, 200, 200)
        ohlc = pd.DataFrame({
            "open": base, "high": base + 2, "low": base - 2, "close": base,
        }, index=idx)
        sig = OvernightReturnFactor(5, 0).compute(ohlc)
        assert sig["overnight"].iloc[-1] == 1.0

    def test_intraday(self):
        ohlc = _mk_ohlc()
        sig = IntradayReturnFactor(5, 0).compute(ohlc)
        assert sig["intraday"].iloc[-1] == 1.0  # close > open

    def test_gap(self):
        ohlc = _mk_ohlc()
        sig = GapFactor(5, 0.002).compute(ohlc)
        assert set(sig["gap"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_close_location(self):
        # close 靠近 high → CLV 高 → 多头
        ohlc = _mk_ohlc()
        ohlc["close"] = ohlc["high"] - 0.1  # 收盘贴高
        sig = CloseLocationFactor(5, 0.5).compute(ohlc)
        assert sig["clv"].iloc[-1] == 1.0

    def test_divergence(self):
        ohlc = _mk_ohlc()
        sig = OvernightIntradayDivergenceFactor(5).compute(ohlc)
        assert set(sig["divergence"].unique()).issubset({0.0, 1.0})


class TestComposite:
    def test_multi_horizon_mom(self):
        sig = MultiHorizonMomentumFactor([63, 126], [0.5, 0.5]).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] == 1.0  # 上行 → 多

    def test_sharpe_mom(self):
        sig = SharpeMomentumFactor(63, 20, 0.5).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_momentum_crash(self):
        # 持续上涨: 短期与长期动量均正 → 无崩溃信号
        sig = MomentumCrashFilterFactor(20, 63).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] == 0.0

    def test_momentum_crash_triggers(self):
        # 冲高回落: 短期转负, 长期仍正 → 崩溃信号
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        ramp = np.linspace(0, 0.4, 170)
        crash = np.linspace(0.4, 0.1, 30)  # 后期急跌
        s = pd.Series(100 * np.exp(np.concatenate([ramp, crash])), index=idx)
        sig = MomentumCrashFilterFactor(20, 63).compute(pd.DataFrame({"A": s}))
        assert (sig["A"] < 0).any()
