# coding=utf-8
"""CTA 新增量价因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.indicators import (
    adx,
    atr,
    bollinger_bands,
    bollinger_percent_b,
    donchian,
    efficiency_ratio,
    rsi,
    zscore,
)
from QuantNodes.strategy.cta.factors.trend import (
    ADXFilterFactor,
    BollingerFactor,
    EfficiencyFactor,
    MAFactor,
    RSIFactor,
    TSMAFactor,
    compute_atr,
)
from QuantNodes.strategy.cta.factors.volatility import (
    ParkinsonFactor,
    VolRegimeFactor,
)
from QuantNodes.strategy.cta.factors.volume import (
    OBVFactor,
    OpenInterestTrendFactor,
    VolumeBreakoutFactor,
    VolumeTrendFactor,
)
from QuantNodes.strategy.cta.factors.momentum_xs import (
    BasisXSFactor,
    MomentumXSFactor,
    VolatilityXSFactor,
)
from QuantNodes.strategy.cta.factors.base import time_series_factor


def _mk_uptrend(n=100, start=100.0, end=200.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.linspace(start, end, n)}, index=idx)


def _mk_ohlc(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    base = np.linspace(100, 200, n)
    return pd.DataFrame({
        "high": base + 2, "low": base - 2, "close": base,
    }, index=idx)


class TestIndicators:
    def test_rsi_uptrend_high(self):
        s = _mk_uptrend()["A"]
        r = rsi(s, 14)
        assert r.iloc[-1] > 60  # 强上行 → RSI 高位

    def test_rsi_bounds(self):
        s = _mk_uptrend()["A"]
        r = rsi(s, 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_atr_positive(self):
        a = atr(_mk_ohlc(), 14)
        assert (a.dropna() > 0).all()

    def test_adx_bounds(self):
        a = adx(_mk_ohlc(), 14).dropna()
        # 强趋势 ADX 可能略超 100 (浮点误差)
        assert (a >= 0).all() and (a <= 100.01).all()

    def test_bollinger_bands_order(self):
        mid, up, lo = bollinger_bands(_mk_uptrend()["A"], 20)
        df = pd.DataFrame({"mid": mid, "up": up, "lo": lo}).dropna()
        assert (df["up"] >= df["mid"]).all()
        assert (df["lo"] <= df["mid"]).all()

    def test_bollinger_percent_b(self):
        pb = bollinger_percent_b(_mk_uptrend()["A"], 20)
        assert pb.dropna().between(0, 2).all()  # 强上行 %B 高

    def test_efficiency_ratio(self):
        er = efficiency_ratio(_mk_uptrend()["A"], 10)
        assert er.dropna().between(0, 1).all()

    def test_zscore(self):
        s = _mk_uptrend()["A"]
        z = zscore(s, 20)
        assert z.dropna().abs().max() > 1  # 上行趋势 z 显著


class TestTrendFactors:
    def test_rsi_factor(self):
        sig = RSIFactor(14, 30, 70).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] <= 0  # 强上行 RSI>70 → 超买信号

    def test_bollinger_factor(self):
        # 区间震荡后突破上轨 → %B>1 → 多头
        idx = pd.date_range("2025-01-01", periods=120, freq="B")
        noise = np.random.RandomState(42).normal(0, 1, 120)
        flat = 100 + noise[:100]
        breakout = np.linspace(100, 130, 20)
        s = pd.Series(np.concatenate([flat, breakout]), index=idx)
        sig = BollingerFactor(20, 2).compute(pd.DataFrame({"A": s}))
        # 突破上轨期间应触发多头
        assert (sig["A"] > 0).sum() >= 5

    def test_efficiency_factor(self):
        sig = EfficiencyFactor(10, 0.5, 0.15).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] == 1.0  # 强趋势 → 多头

    def test_adx_filter(self):
        ohlc = _mk_ohlc()
        f = ADXFilterFactor(14)
        out = f.compute(ohlc)
        assert "adx" in out.columns
        assert out["adx"].iloc[-1] > 0


class TestVolatilityFactors:
    def test_parkinson(self):
        ohlc = _mk_ohlc()
        p = ParkinsonFactor(20).compute(ohlc)
        assert p.dropna().iloc[0] > 0

    def test_parkinson_requires_ohlc(self):
        import pytest
        with pytest.raises(ValueError):
            ParkinsonFactor(20).compute(_mk_uptrend())

    def test_vol_regime(self):
        sig = VolRegimeFactor(20, 60, 1.0).compute(_mk_uptrend())
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestVolumeFactors:
    def _mk_panels(self, n=100):
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
        volumes = pd.DataFrame({"A": np.linspace(100, 400, n)}, index=idx)
        oi = pd.DataFrame({"A": np.linspace(1000, 2000, n)}, index=idx)
        return prices, volumes, oi

    def test_volume_trend(self):
        prices, volumes, _ = self._mk_panels()
        sig = VolumeTrendFactor(20, 5).compute(prices, volumes)
        assert sig["A"].iloc[-1] == 1.0  # 价涨量增 → +1

    def test_volume_breakout(self):
        # 量: 100 天平稳 + 最后 5 天放量 5 倍
        idx = pd.date_range("2025-01-01", periods=105, freq="B")
        vol = np.concatenate([np.full(100, 100.0), np.full(5, 500.0)])
        volumes = pd.DataFrame({"A": vol}, index=idx)
        sig = VolumeBreakoutFactor(20, 2.0).compute(volumes)
        assert sig["A"].iloc[-1] == 1.0  # 放量 → 触发

    def test_oi_trend(self):
        _, _, oi = self._mk_panels()
        sig = OpenInterestTrendFactor(5).compute(oi)
        assert sig["A"].iloc[-1] == 1.0  # 增仓

    def test_obv(self):
        prices, volumes, _ = self._mk_panels()
        sig = OBVFactor(20).compute(prices, volumes)
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 1.0})


class TestXSFactors:
    def _mk_panel(self, n=100):
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        a = np.linspace(100, 300, n)     # 强动量
        b = np.linspace(100, 120, n)     # 弱动量
        c = np.linspace(100, 100, n)     # 无动量
        d = np.linspace(200, 100, n)     # 负动量
        return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)

    def test_momentum_xs(self):
        prices = self._mk_panel()
        sig = MomentumXSFactor(60, 0.25).compute(prices)
        last = sig.iloc[-1]
        assert last["A"] == 1.0   # 最强 → 多
        assert last["D"] == -1.0  # 最弱 → 空
        assert last["B"] == 0.0
        assert last["C"] == 0.0

    def test_vol_xs(self):
        prices = self._mk_panel()
        sig = VolatilityXSFactor(20, 0.25).compute(prices)
        # C 平稳低波 → 多
        assert sig.iloc[-1]["C"] == 1.0

    def test_basis_xs(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="B")
        basis = pd.DataFrame({
            "A": [5.0] * 5,     # 贴水深 (近月 >> 次月)
            "B": [1.0] * 5,
            "C": [-1.0] * 5,
            "D": [-5.0] * 5,    # 升水深
        }, index=idx)
        sig = BasisXSFactor(0.25).compute(basis)
        last = sig.iloc[-1]
        assert last["A"] == 1.0
        assert last["D"] == -1.0


class TestTSMA:
    def test_uptrend_positive(self):
        sig = TSMAFactor(60).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] == 1.0  # 强上行 → 多

    def test_downtrend_negative(self):
        sig = TSMAFactor(60).compute(_mk_uptrend(end=80.0))
        assert sig["A"].iloc[-1] == -1.0  # 下行 → 空

    def test_output_values(self):
        sig = TSMAFactor(20).compute(_mk_uptrend())
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 1.0})


class TestMAFactor:
    def test_cross_direction(self):
        # 短均线 > 长均线 → 多
        sig = MAFactor(3, 10).compute(_mk_uptrend())
        assert sig["A"].iloc[-1] == 1.0

    def test_down_cross(self):
        sig = MAFactor(3, 10).compute(_mk_uptrend(end=50.0))
        assert sig["A"].iloc[-1] == -1.0


class TestIndicatorsExtras:
    def test_donchian(self):
        up, dn = donchian(_mk_uptrend()["A"], 20)
        assert (up.dropna() > dn.dropna()).all()  # 上轨 > 下轨

    def test_donchian_excludes_current(self):
        s = _mk_uptrend()["A"]
        up, _ = donchian(s, 20)
        # 上轨不含当日: 当日价 > 前20日最高 (上行时)
        assert s.iloc[-1] > up.iloc[-1]

    def test_compute_atr(self):
        df = _mk_ohlc()
        a = compute_atr(df, 14)
        assert (a.dropna() > 0).all()


class TestBase:
    def test_time_series_factor(self):
        prices = _mk_uptrend()
        out = time_series_factor(prices, "A", lambda s: s.pct_change(5))
        assert out is not None
        assert len(out) == len(prices)
        assert out.iloc[-1] > 0  # 上行 → 正收益
