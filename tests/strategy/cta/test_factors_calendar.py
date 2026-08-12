# coding=utf-8
"""CTA 四轮因子 (日历/资金/持仓/背离/相关) 测试."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.factors.calendar import (
    MonthTurnFactor,
    SeasonalityFactor,
    WeekdayFactor,
)
from QuantNodes.strategy.cta.factors.capital import (
    CapitalInflowFactor,
    TurnoverFactor,
)
from QuantNodes.strategy.cta.factors.position import (
    FrontMonthOIShareFactor,
    OIConcentrationFactor,
)
from QuantNodes.strategy.cta.factors.divergence import (
    PriceVolumeDivergenceFactor,
    VolumeConfirmationFactor,
)
from QuantNodes.strategy.cta.factors.correlation import (
    AvgCorrelationFactor,
    DispersionFactor,
)


def _mk_panel(n=300, seed=42):
    """多品种随机面板."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    out = {}
    for p in ["A", "B", "C"]:
        out[p] = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame(out, index=idx)


def _mk_vol_oi(price_df):
    rng = np.random.RandomState(7)
    vols = pd.DataFrame(
        {c: rng.randint(1000, 5000, len(price_df)) for c in price_df.columns},
        index=price_df.index,
    )
    ois = pd.DataFrame(
        {c: rng.randint(5000, 20000, len(price_df)) for c in price_df.columns},
        index=price_df.index,
    )
    return vols, ois


class TestCalendar:
    def test_seasonality(self):
        prices = _mk_panel()
        sig = SeasonalityFactor(min_years=1, threshold=0).compute(prices)
        assert sig.shape == prices.shape
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_weekday(self):
        prices = _mk_panel()
        sig = WeekdayFactor(min_weeks=20, threshold=0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_month_turn(self):
        prices = _mk_panel()
        sig = MonthTurnFactor(window=2, threshold=0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestCapital:
    def test_capital_inflow(self):
        prices = _mk_panel()
        vols, ois = _mk_vol_oi(prices)
        sig = CapitalInflowFactor(20, 1.0).compute(prices, ois)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_turnover(self):
        prices = _mk_panel()
        vols, _ = _mk_vol_oi(prices)
        sig = TurnoverFactor(20, 1.5).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestPosition:
    def _mk_spread(self, n=100):
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        rng = np.random.RandomState(3)
        rows = []
        for d in idx:
            for prod in ["AU", "RB"]:
                oi_near = rng.uniform(500, 1000)
                rows.append({
                    "trade_date": d, "product": prod,
                    "oi_near": oi_near,
                    "oi_near2": oi_near * 0.6,
                    "oi_near3": oi_near * 0.3,
                    "oi_near4": oi_near * 0.1,
                })
        return pd.DataFrame(rows)

    def test_oi_concentration(self):
        spread = self._mk_spread()
        sig = OIConcentrationFactor(20, 1.0).compute(spread)
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_front_month_share(self):
        spread = self._mk_spread()
        sig = FrontMonthOIShareFactor(20, 1.0).compute(spread)
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_missing_col(self):
        with pytest.raises(ValueError):
            OIConcentrationFactor().compute(pd.DataFrame({"a": [1]}))


class TestDivergence:
    def test_volume_confirmation(self):
        prices = _mk_panel()
        vols, _ = _mk_vol_oi(prices)
        sig = VolumeConfirmationFactor(20, 1.5).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_pv_divergence(self):
        prices = _mk_panel()
        vols, _ = _mk_vol_oi(prices)
        sig = PriceVolumeDivergenceFactor(20).compute(prices, vols)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestCorrelation:
    def test_avg_correlation(self):
        prices = _mk_panel()
        sig = AvgCorrelationFactor(60, 0.5).compute(prices)
        assert set(sig["avg_corr"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_dispersion(self):
        prices = _mk_panel()
        sig = DispersionFactor(20, 1.0).compute(prices)
        assert set(sig["dispersion"].unique()).issubset({-1.0, 0.0, 1.0})
