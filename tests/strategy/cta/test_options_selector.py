# coding=utf-8
"""期权选择器 + 对冲框架测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.hedge_layer import (
    HedgeConfig, compute_hedge_targets, iv_coefficient,
)
from QuantNodes.strategy.cta.options.selector import select_call, select_put
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader


class TestSelector:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_put_rb(self):
        """RB 选虚值 Put: delta ≈ 目标, strike < 标的."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        sel = select_put("RB", "2025-01-10", px, target_delta=0.25,
                         loader=loader, um=um, oe=oe)
        assert sel is not None
        assert sel.option_type == "P"
        assert sel.strike < px
        assert 0.1 < abs(sel.delta) < 0.5
        assert 20 <= sel.days_to_expiry <= 90

    def test_select_call_rb(self):
        """RB 选虚值 Call: delta > 0, strike > 标的."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        sel = select_call("RB", "2025-01-10", px, target_delta=0.30,
                          loader=loader, um=um, oe=oe)
        assert sel is not None
        assert sel.option_type == "C"
        assert sel.strike > px
        assert sel.delta > 0.1

    def test_select_put_nontrading_day(self):
        """非交易日 (周末) → None."""
        loader, um, oe = self._setup()
        sel = select_put("M", "2026-01-10", 2786.0,  # 周六
                         loader=loader, um=um, oe=oe)
        assert sel is None


class TestHedgeLayer:
    def test_iv_coefficient_low(self):
        cfg = HedgeConfig()
        assert iv_coefficient(0.1, cfg) == 1.0

    def test_iv_coefficient_mid(self):
        cfg = HedgeConfig()
        assert iv_coefficient(0.5, cfg) == 0.5

    def test_iv_coefficient_high(self):
        cfg = HedgeConfig()
        assert iv_coefficient(0.9, cfg) == 0.0

    def test_iv_coefficient_nan(self):
        cfg = HedgeConfig()
        assert iv_coefficient(np.nan, cfg) == 1.0

    def test_compute_hedge_targets_placeholder(self):
        """框架占位: 当前返回空 dict (策略后延)."""
        out = compute_hedge_targets(
            cta_positions=pd.DataFrame(),
            iv_panel=pd.DataFrame(),
            prices=pd.DataFrame(),
            config=HedgeConfig(),
        )
        assert out == {}
