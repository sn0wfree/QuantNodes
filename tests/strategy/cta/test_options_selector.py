# coding=utf-8
"""期权选择器 + 对冲框架测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.hedge_layer import (
    HedgeConfig, basis_coefficient, compute_hedge_targets, iv_coefficient,
    timing_coefficient,
)
from QuantNodes.strategy.cta.options.selector import (
    OptionSelection, select_call, select_put,
)
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

    def test_basis_coefficient_deep_back(self):
        """深贴水 (分位高) → 全额买."""
        cfg = HedgeConfig()
        assert basis_coefficient(0.90, cfg) == 1.0

    def test_basis_coefficient_mid(self):
        cfg = HedgeConfig()
        assert basis_coefficient(0.50, cfg) == 0.5

    def test_basis_coefficient_contango(self):
        """深升水 (分位低) → 不买."""
        cfg = HedgeConfig()
        assert basis_coefficient(0.10, cfg) == 0.0

    def test_basis_coefficient_nan(self):
        cfg = HedgeConfig()
        assert basis_coefficient(np.nan, cfg) == 1.0

    def test_timing_none(self):
        cfg = HedgeConfig(timing_mode="none")
        assert timing_coefficient(0.9, 0.1, cfg) == 1.0

    def test_timing_both_min(self):
        """both: iv 全额但 basis 不买 → 0."""
        cfg = HedgeConfig(timing_mode="both")
        assert timing_coefficient(0.1, 0.1, cfg) == 0.0

    def test_timing_both_allow(self):
        cfg = HedgeConfig(timing_mode="both")
        assert timing_coefficient(0.1, 0.9, cfg) == 1.0

    def test_compute_empty(self):
        out = compute_hedge_targets(pd.DataFrame(), pd.DataFrame(),
                                    pd.DataFrame(), config=HedgeConfig())
        assert out == {}

    def test_compute_short_not_hedged(self):
        """净空头 → 不买 put (MVP 单边保护)."""
        idx = pd.DatetimeIndex(["2025-01-10"])
        cta = pd.DataFrame({"RB": [-5.0]}, index=idx)
        iv = pd.DataFrame({"RB": [0.1]}, index=idx)
        px = pd.DataFrame({"RB": [3200.0]}, index=idx)
        out = compute_hedge_targets(cta, iv, px, config=HedgeConfig())
        assert out == {}

    def test_compute_uses_select_fun(self):
        """注入 select_fun 验证数量计算 (预算约束)."""
        idx = pd.DatetimeIndex(["2025-01-10"])
        cta = pd.DataFrame({"RB": [10.0]}, index=idx)
        iv = pd.DataFrame({"RB": [0.1]}, index=idx)
        px = pd.DataFrame({"RB": [3200.0]}, index=idx)

        def fake_select(product, ts, price, **kw):
            return OptionSelection(
                product="RB", symbol="rb2505P3100", option_type="P",
                strike=3100.0, underlying_price=3200.0, delta=-0.25,
                premium=30.0, expiry="2025-03-10", days_to_expiry=40,
            )

        cfg = HedgeConfig(timing_mode="iv")
        out = compute_hedge_targets(cta, iv, px, config=cfg,
                                    select_fun=fake_select)
        # delta 对冲: n = qty × ratio / |delta| = 10 × 0.4 / 0.25 = 16
        # 预算约束: budget = 0.005 × 1M = 5000; max = 5000/(30×1) = 166; min(16,166)=16
        assert out == {"rb2505P3100": 16.0}

    def test_compute_iv_high_blocks(self):
        """IV 分位高 → 不买."""
        idx = pd.DatetimeIndex(["2025-01-10"])
        cta = pd.DataFrame({"RB": [10.0]}, index=idx)
        iv = pd.DataFrame({"RB": [0.9]}, index=idx)
        px = pd.DataFrame({"RB": [3200.0]}, index=idx)

        def fake_select(product, ts, price, **kw):
            return OptionSelection(
                product="RB", symbol="rb2505P3100", option_type="P",
                strike=3100.0, underlying_price=3200.0, delta=-0.25,
                premium=30.0, expiry="2025-03-10", days_to_expiry=40,
            )

        out = compute_hedge_targets(cta, iv, px, config=HedgeConfig(),
                                    select_fun=fake_select)
        assert out == {}

    def test_compute_basis_contango_blocks(self):
        """basis 模式: 深升水 → 不买."""
        idx = pd.DatetimeIndex(["2025-01-10"])
        cta = pd.DataFrame({"RB": [10.0]}, index=idx)
        iv = pd.DataFrame({"RB": [np.nan]}, index=idx)
        px = pd.DataFrame({"RB": [3200.0]}, index=idx)
        basis = pd.DataFrame({"RB": [0.1]}, index=idx)

        def fake_select(product, ts, price, **kw):
            return OptionSelection(
                product="RB", symbol="rb2505P3100", option_type="P",
                strike=3100.0, underlying_price=3200.0, delta=-0.25,
                premium=30.0, expiry="2025-03-10", days_to_expiry=40,
            )

        out = compute_hedge_targets(cta, iv, px, config=HedgeConfig(timing_mode="basis"),
                                    basis_panel=basis, select_fun=fake_select)
        assert out == {}

    def test_compute_min_notional(self):
        """低于 min_net_notional → 不买."""
        idx = pd.DatetimeIndex(["2025-01-10"])
        cta = pd.DataFrame({"RB": [1.0]}, index=idx)
        iv = pd.DataFrame({"RB": [0.1]}, index=idx)
        px = pd.DataFrame({"RB": [3200.0]}, index=idx)

        def fake_select(product, ts, price, **kw):
            return OptionSelection(
                product="RB", symbol="rb2505P3100", option_type="P",
                strike=3100.0, underlying_price=3200.0, delta=-0.25,
                premium=30.0, expiry="2025-03-10", days_to_expiry=40,
            )

        cfg = HedgeConfig(min_net_notional=1e7)
        out = compute_hedge_targets(cta, iv, px, config=cfg,
                                    select_fun=fake_select)
        assert out == {}
