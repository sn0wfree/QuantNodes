# coding=utf-8
"""宏观联动与事件因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.macro_link import (
    DXYLinkFactor,
    MacroStateFactor,
    VIXLinkFactor,
)
from QuantNodes.strategy.cta.factors.event import (
    ExtremeMoveFactor,
    GapStreakFactor,
    LimitMoveFactor,
)


def _mk_prices(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _mk_vix(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    vix = np.concatenate([np.full(n // 2, 15.0), np.linspace(15, 35, n - n // 2)])
    return pd.Series(vix, index=idx)


def _mk_dxy(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    dxy = np.concatenate([np.linspace(100, 105, n // 2), np.linspace(105, 100, n - n // 2)])
    return pd.Series(dxy, index=idx)


class TestVIXLink:
    def test_vix_rise_short(self):
        prices = _mk_prices()
        vix = _mk_vix()
        # 阈值 0.5 < VIX 5日变化 0.78
        sig = VIXLinkFactor(5, 0.5).compute(prices, vix)
        # VIX 后半段大涨 → 空头信号
        assert (sig.iloc[200:] == -1.0).any().any()

    def test_output_values(self):
        prices = _mk_prices()
        vix = _mk_vix()
        sig = VIXLinkFactor(5, 0.5).compute(prices, vix)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestDXYLink:
    def test_dxy_rise_short(self):
        prices = _mk_prices()
        dxy = _mk_dxy()
        # 阈值 0.15 < DXY 5日变化 0.19
        sig = DXYLinkFactor(5, 0.15).compute(prices, dxy)
        # DXY 前半段上升 → 空头
        assert (sig.iloc[20:100] == -1.0).any().any()

    def test_output_values(self):
        prices = _mk_prices()
        dxy = _mk_dxy()
        sig = DXYLinkFactor(5, 0.15).compute(prices, dxy)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestMacroState:
    def test_risk_off(self):
        prices = _mk_prices()
        vix = _mk_vix()  # 后半段 VIX 25+ 且上升 → risk-off
        sig = MacroStateFactor(20, 25.0, 5).compute(prices, vix)
        assert (sig == -1.0).any().any()

    def test_output_values(self):
        prices = _mk_prices()
        vix = _mk_vix()
        sig = MacroStateFactor(20, 25.0, 5).compute(prices, vix)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestExtremeMove:
    def test_extreme_up(self):
        # 平稳后单日暴涨
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.RandomState(1)
        rets = rng.normal(0.0005, 0.005, n)
        rets[-1] = 0.06  # 极端上涨
        prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)
        sig = ExtremeMoveFactor(20, 3.0, "both").compute(prices)
        assert sig["A"].iloc[-1] == 1.0

    def test_output_values(self):
        prices = _mk_prices()
        sig = ExtremeMoveFactor(20, 3.0, "both").compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestGapStreak:
    def test_streak_detection(self):
        # 连续 4 天上涨
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 140, n)}, index=idx)
        sig = GapStreakFactor("up", 3).compute(prices)
        assert (sig["A"] == 1.0).sum() >= 3

    def test_output_values(self):
        prices = _mk_prices()
        sig = GapStreakFactor("both", 3).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestLimitMove:
    def test_limit_up(self):
        # 单日 +6% 接近涨跌停
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.RandomState(2)
        rets = rng.normal(0, 0.005, n)
        rets[-1] = 0.055
        prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rets))}, index=idx)
        sig = LimitMoveFactor(0.05, 0.005).compute(prices)
        assert sig["A"].iloc[-1] == 1.0

    def test_no_signal_quiet(self):
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 105, n)}, index=idx)
        sig = LimitMoveFactor(0.05, 0.005).compute(prices)
        assert (sig["A"] == 0).all()
