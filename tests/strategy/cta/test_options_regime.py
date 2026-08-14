# coding=utf-8
"""Regime Adaptive 策略测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.strategies import get_strategy, list_strategies
from QuantNodes.strategy.cta.options.strategies.regime_adaptive import RegimeAdaptive
from QuantNodes.strategy.cta.options.vol_signals import Regime, VolSignalGenerator
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader


class TestRegimeAdaptiveRegistry:
    def test_registered(self):
        assert "regime_adaptive" in list_strategies()

    def test_get(self):
        s = get_strategy("regime_adaptive")
        assert isinstance(s, RegimeAdaptive)


class TestRegimeAdaptive:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def _make_vol_gen(self, product="RB", iv_high=False, hv_high=False):
        """创建 mock VolSignalGenerator."""
        n = 30
        if iv_high:
            iv_vals = [0.10] * 10 + list(np.linspace(0.10, 0.25, 20))
        else:
            iv_vals = [0.12] * n

        if hv_high:
            hv_vals = [0.08] * 10 + list(np.linspace(0.08, 0.30, 20))
        else:
            hv_vals = [0.08] * n

        iv_panel = pd.DataFrame({product: iv_vals},
                                index=pd.date_range("2025-01-01", periods=n))
        hv_panel = pd.DataFrame({product: hv_vals},
                                index=pd.date_range("2025-01-01", periods=n))
        return VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=10)

    def test_low_vol_no_drop_routes_to_put_spread(self):
        """低波不跌 → 卖 put spread."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        vol_gen = self._make_vol_gen(iv_high=False, hv_high=False)

        s = get_strategy("regime_adaptive")
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe, vol_signal_gen=vol_gen)
        # 卖 put spread: 买高 strike + 卖低 strike
        if targets:
            has_buy = any(v > 0 for v in targets.values())
            has_sell = any(v < 0 for v in targets.values())
            assert has_buy and has_sell

    def test_high_vol_no_drop_routes_to_strangle(self):
        """高波不跌 → 卖 strangle."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-20")
        vol_gen = self._make_vol_gen(iv_high=True, hv_high=False)

        s = get_strategy("regime_adaptive")
        targets = s.select("RB", pd.Timestamp("2025-01-20"), px, None,
                           loader=loader, um=um, oe=oe, vol_signal_gen=vol_gen)
        if targets:
            # 卖 strangle: 卖 call + 卖 put
            assert all(v < 0 for v in targets.values())

    def test_no_entry_when_holding(self):
        """有持仓时不入场."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        vol_gen = self._make_vol_gen()

        s = get_strategy("regime_adaptive")
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px,
                           {"type": "put_spread"}, loader=loader, um=um, oe=oe,
                           vol_signal_gen=vol_gen)
        assert targets == {}

    def test_no_signal_gen_uses_default(self):
        """无 VolSignalGenerator 时使用默认 (不操作)."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")

        s = get_strategy("regime_adaptive")
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        assert targets == {}
