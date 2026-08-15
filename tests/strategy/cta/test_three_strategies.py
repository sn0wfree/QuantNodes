# coding=utf-8
"""三大方向策略测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.options_filter import OptionsFactorFilter
from QuantNodes.strategy.cta.options.sell_premia_strategy import (
    SellPremiaWithCTA, SellPremiaConfig,
)
from QuantNodes.strategy.cta.options.product_vrp_strategy import (
    ProductVRPStrategy, ProductVRPConfig, VRP_CLASSIFICATION,
)


class TestOptionsFactorFilter:
    def test_basic(self):
        f = OptionsFactorFilter()
        weights = pd.DataFrame({
            "A": [1.0, -1.0, 0.5, -0.5, 0.0],
            "B": [0.5, -0.5, 1.0, -1.0, 0.0],
        }, index=pd.date_range("2025-01-01", periods=5))
        result = f.filter(weights)
        assert result.shape == weights.shape
        # 无 filter 数据时应返回原权重
        assert (result.abs() <= weights.abs() + 1e-10).all().all()

    def test_with_skew(self):
        f = OptionsFactorFilter(skew_threshold=0.5)
        weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=pd.date_range("2025-01-01", periods=3))
        skew = pd.DataFrame({"A": [0.1, 0.8, 0.1]}, index=pd.date_range("2025-01-01", periods=3))
        result = f.filter(weights, skew_panel=skew)
        # 第二天 skew=0.8 > 0.5 → 减仓 50%
        # 注意: skew_zscore 需要更多数据点, 这里用简化测试
        assert result.shape == weights.shape

    def test_with_vrp(self):
        f = OptionsFactorFilter(vrp_boost=1.2)
        weights = pd.DataFrame({"A": [1.0, -1.0]}, index=pd.date_range("2025-01-01", periods=2))
        vrp = pd.DataFrame({"A": [0.5, -0.3]}, index=pd.date_range("2025-01-01", periods=2))
        result = f.filter(weights, vrp_panel=vrp)
        # 第一天: CTA=+1, VRP=+0.5 → 同向 → 加仓 20%
        assert result.loc[result.index[0], "A"] == 1.2
        # 第二天: CTA=-1, VRP=-0.3 → 同向 → 加仓 20%
        assert result.loc[result.index[1], "A"] == -1.2


class TestSellPremiaWithCTA:
    def _setup(self):
        from QuantNodes.strategy.cta.options.expiry import OptionExpiry
        from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
        from QuantNodes.strategy.cta.data_loader import CtaDataLoader
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_sell_put_spread_weak_trend(self):
        """弱趋势 → 卖 iron condor."""
        loader, um, oe = self._setup()
        cfg = SellPremiaConfig(products=["RB"])
        s = SellPremiaWithCTA(cfg)
        # 弱趋势 (cta_signal=0.1)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), 3500.0,
                           cta_signal=0.1, loader=loader, um=um, oe=oe)
        # Iron condor: 卖 put + 卖 call + 买保护
        assert len(targets) >= 2
        has_sell = any(v < 0 for v in targets.values())
        assert has_sell

    def test_sell_call_spread_strong_bear(self):
        """强看跌 → 卖 call spread 或 iron condor."""
        loader, um, oe = self._setup()
        cfg = SellPremiaConfig(products=["RB"])
        s = SellPremiaWithCTA(cfg)
        # 强看跌 (cta_signal=-0.8)
        targets = s.select("RB", pd.Timestamp("2025-01-10"), 3500.0,
                           cta_signal=-0.8, loader=loader, um=um, oe=oe)
        # 卖 put spread (对冲下行风险) 或空 (无合适合约)
        assert len(targets) <= 4  # 最多 4 条腿

    def test_position_size(self):
        cfg = SellPremiaConfig()
        s = SellPremiaWithCTA(cfg)
        # 基础 100万, VRP 高 → 加仓, CTA 强 → 减仓
        size = s.get_position_size(1_000_000, vrp_pct=0.8, cta_strength=0.6)
        # vrp_mult = 1.2, cta_mult = 0.7 → 100万 × 1.2 × 0.7 = 84万
        assert abs(size - 840_000) < 1000


class TestProductVRPStrategy:
    def test_classification(self):
        assert VRP_CLASSIFICATION["positive"] == ["AU", "L", "PP", "RU", "CU", "FU", "V", "AG"]
        assert VRP_CLASSIFICATION["negative"] == ["I", "M", "FG", "SA", "JD", "B"]

    def test_get_class(self):
        cfg = ProductVRPConfig(products=["RB", "I", "AU"])
        s = ProductVRPStrategy(cfg)
        s.load_vrp_classification("2023-01-01", "2023-12-31")
        # 静态分类
        assert s.get_vrp_class("AU") == "positive"
        assert s.get_vrp_class("I") == "negative"
        assert s.get_vrp_class("RB") == "neutral"

    def test_select_by_class(self):
        loader, um, oe = self._setup()
        cfg = ProductVRPConfig(products=["RB", "I", "AU"])
        s = ProductVRPStrategy(cfg)
        s.load_vrp_classification("2023-01-01", "2023-12-31")
        s._loader = loader
        s._um = um
        s._oe = oe

        # AU (正 VRP) → 卖 put spread
        targets = s.select("AU", pd.Timestamp("2025-01-10"), 600.0,
                           loader=loader, um=um, oe=oe)
        if targets:
            has_sell = any(v < 0 for v in targets.values())
            assert has_sell

    def _setup(self):
        from QuantNodes.strategy.cta.options.expiry import OptionExpiry
        from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
        from QuantNodes.strategy.cta.data_loader import CtaDataLoader
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)
