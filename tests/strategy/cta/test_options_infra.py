# coding=utf-8
"""期权基础设施测试: underlying_map / expiry / iv_panel."""
import pandas as pd

from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.iv_panel import add_percentile, load_iv_panel
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap


class TestUnderlyingMap:
    def test_get_price_series(self):
        um = UnderlyingMap(CtaDataLoader())
        s = um.get("m2608")
        assert len(s) > 0
        assert s.iloc[-1] > 0

    def test_get_unknown_returns_empty(self):
        um = UnderlyingMap(CtaDataLoader())
        s = um.get("XX9999")
        assert len(s) == 0

    def test_get_at(self):
        um = UnderlyingMap(CtaDataLoader())
        px = um.get_at("m2608", "2026-03-10")
        assert px is None or px > 0


class TestOptionExpiry:
    def test_get_expiry_dce(self):
        oe = OptionExpiry(CtaDataLoader())
        d = oe.get("m2608-P-2750")
        assert d is not None
        assert d.year == 2026 and d.month == 3  # DCE m2608 3 月到期

    def test_get_expiry_shfe_undashed(self):
        """SHFE 无横线格式 rb2303C3250 → 产品 RB."""
        oe = OptionExpiry(CtaDataLoader())
        d = oe.get("rb2303C3250")
        assert d is not None
        assert d.year == 2023 and d.month == 2  # RB 期权 2 月末到期

    def test_get_unknown_none(self):
        oe = OptionExpiry(CtaDataLoader())
        assert oe.get("XX9999-C-100") is None


class TestIVPanel:
    def test_add_percentile(self):
        panel = pd.DataFrame(
            {c: pd.Series(range(300), index=pd.date_range("2020-01-01", periods=300))
             for c in ["M", "RB"]},
        )
        pct = add_percentile(panel)
        # 后期分位 ∈ [0,1]
        tail = pct.iloc[-50:]
        assert tail.min().min() >= 0.0
        assert tail.max().max() <= 1.0
        # 单调序列的 rank 应近似线性上升
        mid = pct["M"].iloc[-50:].mean()
        assert 0.9 < mid <= 1.0

    def test_load_iv_panel_table(self):
        """option_iv_daily 表可读 (需先 build_iv_panel + save_iv_panel)."""
        try:
            panel = load_iv_panel(products=["M"])
        except Exception:
            import pytest
            pytest.skip("option_iv_daily 未构建, 先跑 build_iv_panel.py")
        assert not panel.empty
        assert "M" in panel.columns
