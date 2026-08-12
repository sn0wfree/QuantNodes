# coding=utf-8
"""CTA 策略测试."""
import pandas as pd

from QuantNodes.strategy.cta.strategies.trend_following import (
    DEFAULT_PRODUCTS,
    TrendFollowingConfig,
    TrendFollowingStrategy,
)
from QuantNodes.strategy.cta.factors.trend import DonchianChannelFactor
from QuantNodes.strategy.cta.factors.volatility import vol_target_positions


class TestTrendFollowingConfig:
    def test_defaults(self):
        cfg = TrendFollowingConfig()
        assert cfg.entry_window == 20
        assert cfg.exit_window == 10
        assert cfg.target_vol == 0.15
        assert cfg.initial_cash == 2_000_000


class TestDefaultProducts:
    def test_nonempty(self):
        assert len(DEFAULT_PRODUCTS) >= 20
        assert "IF" in DEFAULT_PRODUCTS
        assert "AU" in DEFAULT_PRODUCTS


class TestTrendFollowing:
    def test_construct(self):
        strat = TrendFollowingStrategy(
            products=["AU", "RB"], start="2020-01-01", end="2020-06-30",
        )
        assert strat.products == ["AU", "RB"]

    def test_panel_loads(self):
        strat = TrendFollowingStrategy(products=["AU"], start="2020-01-01")
        strat.load_panel()
        assert strat._price_panel is not None
        assert "AU" in strat._price_panel.columns
        assert len(strat._price_panel) > 100

    def test_signals_direction(self):
        strat = TrendFollowingStrategy(products=["AU"], start="2020-01-01")
        strat.load_panel()
        sig = strat._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        # 信号有翻转 (非全程单边)
        assert (sig["AU"].diff().abs() > 0).sum() > 2

    def test_positions_integer(self):
        strat = TrendFollowingStrategy(products=["AU", "RB"], start="2020-01-01")
        strat.load_panel()
        pos = strat._positions()
        # 手数应为整数
        non_zero = pos[pos != 0].stack()
        assert len(non_zero) > 0
        assert all(float(x).is_integer() for x in non_zero)


class TestDonchianIntegration:
    def test_factor_consistent(self):
        import numpy as np
        idx = pd.date_range("2025-01-01", periods=60, freq="B")
        df = pd.DataFrame({"A": np.linspace(100, 150, 60)}, index=idx)
        f = DonchianChannelFactor(10, 5)
        sig = f.compute(df)
        assert sig["A"].iloc[-1] == 1.0

    def test_vol_target_shape(self):
        idx = pd.date_range("2025-01-01", periods=60, freq="B")
        df = pd.DataFrame({"A": pd.Series(range(60, 120), index=idx, dtype=float)}, index=idx)
        sig = pd.DataFrame({"A": [1.0] * 60}, index=idx)
        pos = vol_target_positions(sig, df, target_vol=0.2)
        assert pos.shape == sig.shape
        assert (pos >= 0).all().all()
