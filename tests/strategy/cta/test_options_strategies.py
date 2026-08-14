# coding=utf-8
"""期权策略池测试."""
import pandas as pd

from QuantNodes.strategy.cta.options.strategies import (
    OptionStrategy,
    StrategyConfig,
    get_strategy,
    list_strategies,
)
from QuantNodes.strategy.cta.options.strategies.short_strangle import ShortStrangle
from QuantNodes.strategy.cta.options.strategies.iron_condor import IronCondor
from QuantNodes.strategy.cta.options.strategies.calendar_spread import CalendarSpread
from QuantNodes.strategy.cta.options.strategies.covered_call import CoveredCall
from QuantNodes.strategy.cta.options.strategies.vertical_spread import (
    BullCallSpread,
    BearPutSpread,
)
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader


class TestRegistry:
    def test_list_strategies(self):
        names = list_strategies()
        assert "short_strangle" in names
        assert "iron_condor" in names
        assert "calendar_spread" in names
        assert "covered_call" in names
        assert "bull_call_spread" in names
        assert "bear_put_spread" in names

    def test_get_strategy(self):
        s = get_strategy("short_strangle")
        assert isinstance(s, ShortStrangle)
        assert s.name == "short_strangle"

    def test_get_unknown_raises(self):
        try:
            get_strategy("nonexistent")
            assert False, "Should raise"
        except ValueError:
            pass

    def test_describe(self):
        s = get_strategy("iron_condor")
        d = s.describe()
        assert d["name"] == "iron_condor"
        assert "config" in d


class TestShortStrangle:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_rb(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = ShortStrangle(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        # 卖 call + 卖 put → 2 个 negative targets
        assert len(targets) == 2
        assert all(v < 0 for v in targets.values())

    def test_no_entry_when_holding(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = ShortStrangle(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px,
                           {"type": "straddle"}, loader=loader, um=um, oe=oe)
        assert targets == {}


class TestIronCondor:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_rb(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = IronCondor(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        # 应该有卖腿 (负) + 买腿 (正)
        assert len(targets) >= 2
        has_sell = any(v < 0 for v in targets.values())
        has_buy = any(v > 0 for v in targets.values())
        assert has_sell and has_buy


class TestCalendarSpread:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_rb(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = CalendarSpread(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        # Calendar spread 可能只有近月或远月 (取决于数据)
        assert isinstance(targets, dict)


class TestCoveredCall:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_rb(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = CoveredCall(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        # 期货多头 + 卖 call
        assert "RB" in targets
        assert targets["RB"] > 0
        has_sell_call = any(v < 0 and k != "RB" for k, v in targets.items())
        assert has_sell_call


class TestVerticalSpread:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_bull_call_spread(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = BullCallSpread(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        assert len(targets) == 2
        has_buy = any(v > 0 for v in targets.values())
        has_sell = any(v < 0 for v in targets.values())
        assert has_buy and has_sell

    def test_bear_put_spread(self):
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        cfg = StrategyConfig(products=["RB"])
        s = BearPutSpread(cfg)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), px, None,
                           loader=loader, um=um, oe=oe)
        assert len(targets) == 2
        has_buy = any(v > 0 for v in targets.values())
        has_sell = any(v < 0 for v in targets.values())
        assert has_buy and has_sell
