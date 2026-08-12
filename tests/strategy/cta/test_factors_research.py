# coding=utf-8
"""CTA 研报借鉴因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.term_structure import (
    NearBasisFactor,
    TermCurvatureFactor,
    TermSlopeFactor,
)
from QuantNodes.strategy.cta.factors.volatility import VolSpreadFactor
from QuantNodes.strategy.cta.factors.volume import (
    AmihudFactor,
    VolOIRatioFactor,
    VolumePriceCorrFactor,
)


def _mk_panels(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
    volumes = pd.DataFrame({"A": np.linspace(100, 400, n)}, index=idx)
    oi = pd.DataFrame({"A": np.linspace(1000, 2000, n)}, index=idx)
    return prices, volumes, oi


def _mk_spread(n=5):
    return pd.DataFrame([
        {"trade_date": "2025-01-06", "product": "AU",
         "close_near": 500.0, "close_near2": 502.0, "close_near3": 504.0,
         "close_near4": 508.0},
        {"trade_date": "2025-01-07", "product": "AU",
         "close_near": 505.0, "close_near2": 506.0, "close_near3": 507.0,
         "close_near4": 509.0},
    ] * 3)


class TestVolOI:
    def test_ratio_signal(self):
        prices, volumes, oi = _mk_panels()
        sig = VolOIRatioFactor(20, 1.0).compute(volumes, oi)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})
        assert (sig["A"] != 0).any()


class TestAmihud:
    def test_amihud_signal(self):
        prices, volumes, _ = _mk_panels()
        sig = AmihudFactor(20, 1.0).compute(prices, volumes)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_amihud_reasonable(self):
        # 非流动性 = |ret|/vol, 恒正
        prices, volumes, _ = _mk_panels()
        ret = prices.pct_change().abs()
        illiq = ret / volumes
        assert (illiq.dropna() > 0).all().all()


class TestVolumePriceCorr:
    def test_positive_corr_uptrend(self):
        # 价格与成交量同向波动 → 正相关 → +1
        idx = pd.date_range("2025-01-01", periods=120, freq="B")
        rng = np.random.RandomState(42)
        # 价格随机游走上行
        rets = rng.normal(0.001, 0.01, 120)
        prices = pd.Series(100 * np.exp(rets.cumsum()), index=idx)
        # 成交量与收益同向 → 正相关
        vols = pd.Series(100 + rets * 5000, index=idx)
        p = pd.DataFrame({"A": prices}, index=idx)
        v = pd.DataFrame({"A": vols}, index=idx)
        corr = p.pct_change().rolling(20).corr(v)
        assert (corr.dropna().iloc[-1] > 0).all()  # 正相关
        sig = VolumePriceCorrFactor(20, 0.3).compute(p, v)
        assert (sig["A"] > 0).any()


class TestVolSpread:
    def test_signal_values(self):
        prices, _, _ = _mk_panels()
        sig = VolSpreadFactor(10, 60, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestTermStructure:
    def test_term_slope(self):
        out = TermSlopeFactor().compute(_mk_spread())
        assert len(out) == 6
        # contango: F4 > F1 → slope > 0
        assert out.iloc[0] > 0

    def test_term_curvature(self):
        out = TermCurvatureFactor().compute(_mk_spread())
        assert len(out) == 6
        # F1 - 2F2 + F3 = 500 - 1004 + 504 = 0 → 线性期限结构曲率≈0
        assert abs(out.iloc[0]) < 1e-6

    def test_near_basis(self):
        basis = pd.DataFrame([
            {"trade_date": "2025-01-06", "symbol": "au",
             "near_basis_rate": 0.01},
            {"trade_date": "2025-01-07", "symbol": "au",
             "near_basis_rate": -0.005},
        ])
        out = NearBasisFactor().compute(basis)
        assert out.iloc[0] == 0.01
        # symbol 归一为大写
        assert out.index.get_level_values(1)[0] == "AU"
