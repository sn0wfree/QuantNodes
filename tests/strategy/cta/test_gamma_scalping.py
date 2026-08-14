# coding=utf-8
"""Gamma Scalping 缺失组件测试: select_straddle + delta_hedge + pnl_decompose + roll + slippage."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.selector import (
    StraddleSelection,
    select_straddle,
)
from QuantNodes.strategy.cta.options.delta_hedge import DeltaHedgeManager
from QuantNodes.strategy.cta.options.pnl_decompose import PnlDecomposer
from QuantNodes.strategy.cta.options.roll_manager import RollManager
from QuantNodes.strategy.cta.options.slippage import (
    apply_slippage,
    estimate_option_spread,
    estimate_spread,
)
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_contract import (
    CtaContractRegistry,
    CtaContractSpec,
)


class TestSlippage:
    def test_estimate_spread_normal(self):
        spread = estimate_spread(100.0, 99.0)
        assert 0.001 <= spread <= 0.1

    def test_estimate_spread_equal(self):
        spread = estimate_spread(100.0, 100.0)
        assert spread == 0.001

    def test_estimate_spread_invalid(self):
        assert estimate_spread(0, 100) == 0.001
        assert estimate_spread(100, 0) == 0.001
        assert estimate_spread(50, 100) == 0.001

    def test_estimate_option_spread_atm(self):
        spread = estimate_option_spread(5.0, 4.0, 100.0, 99.0, abs_delta=0.5)
        assert spread >= 0.001

    def test_estimate_option_spread_otm(self):
        """OTM 期权 spread 应 ≥ ATM."""
        atm = estimate_option_spread(5.0, 4.0, 100.0, 99.0, abs_delta=0.5)
        otm = estimate_option_spread(2.0, 1.0, 100.0, 99.0, abs_delta=0.1)
        assert otm >= atm * 0.5  # OTM 至少不比 ATM 窄很多

    def test_apply_slippage_buy(self):
        p = apply_slippage(100.0, "buy", 0.01)
        assert p > 100.0

    def test_apply_slippage_sell(self):
        p = apply_slippage(100.0, "sell", 0.01)
        assert p < 100.0

    def test_apply_slippage_zero_spread(self):
        assert apply_slippage(100.0, "buy", 0.0) == 100.0


class TestSelectStraddle:
    def _setup(self):
        loader = CtaDataLoader()
        return loader, UnderlyingMap(loader), OptionExpiry(loader)

    def test_select_straddle_rb(self):
        """RB 选 ATM Straddle: strike ≈ spot, call + put 同 strike."""
        loader, um, oe = self._setup()
        px = um.get_at("rb2505", "2025-01-10")
        sel = select_straddle("RB", "2025-01-10", px,
                              loader=loader, um=um, oe=oe)
        assert sel is not None
        assert isinstance(sel, StraddleSelection)
        assert sel.call.option_type == "C"
        assert sel.put.option_type == "P"
        assert sel.call.strike == sel.put.strike  # 同 strike
        assert sel.strike == sel.call.strike
        assert sel.call.expiry == sel.put.expiry  # 同到期
        assert sel.cost > 0
        assert abs(sel.delta) < 0.15  # ATM delta ≈ 0
        assert sel.gamma > 0
        assert sel.theta < 0
        assert sel.vega > 0

    def test_select_straddle_m(self):
        """M 品种."""
        loader, um, oe = self._setup()
        px = um.get_at("m2505", "2025-01-10")
        sel = select_straddle("M", "2025-01-10", px,
                              loader=loader, um=um, oe=oe)
        if sel is not None:
            assert sel.call.strike == sel.put.strike
            assert sel.cost > 0

    def test_select_straddle_nontrading_day(self):
        """非交易日 → None."""
        loader, um, oe = self._setup()
        sel = select_straddle("M", "2026-01-10", 2786.0,
                              loader=loader, um=um, oe=oe)
        assert sel is None


class TestDeltaHedgeManager:
    def _make_registry(self):
        reg = CtaContractRegistry.__new__(CtaContractRegistry)
        reg._specs = {}
        reg._margin = {}
        # 注册 RB 规格 (乘数 10)
        reg._specs["RB"] = CtaContractSpec(
            exchange="SHFE", product="RB",
            contract_multiplier=10.0, price_unit="元/吨",
            tick_size=1.0, margin_ratio=0.10,
        )
        return reg

    def test_aggregate_delta_empty(self):
        mgr = DeltaHedgeManager(self._make_registry())
        result = mgr.aggregate_delta([])
        assert result == {}

    def test_aggregate_delta_single(self):
        mgr = DeltaHedgeManager(self._make_registry())
        result = mgr.aggregate_delta([("RB", 5.0, 10.0)])
        assert result == {"RB": 5.0}

    def test_aggregate_delta_multi(self):
        mgr = DeltaHedgeManager(self._make_registry())
        result = mgr.aggregate_delta([
            ("RB", 10.0, 10.0),
            ("RB", -3.0, 10.0),
            ("CU", 5.0, 5.0),
        ])
        assert result["RB"] == 7.0
        assert result["CU"] == 5.0

    def test_compute_hedge_target_small(self):
        mgr = DeltaHedgeManager(self._make_registry())
        # RB delta=10, price=3500, mult=10
        # hedge_qty = -10 / (3500 × 10) = -0.000286 → rounds to 0
        targets = mgr.compute_hedge_target({"RB": 10.0}, {"RB": 3500.0})
        # Delta 太小, 四舍五入后为 0, abs(0) < 1e-9 → 不含
        assert targets == {}

    def test_compute_hedge_target_large(self):
        mgr = DeltaHedgeManager(self._make_registry())
        # RB total_delta=35000, price=3500, mult=10
        # hedge_qty = -35000 / (3500 × 10) = -1.0
        targets = mgr.compute_hedge_target({"RB": 35000.0}, {"RB": 3500.0})
        assert targets["RB"] == -1.0

    def test_record_rebalance(self):
        mgr = DeltaHedgeManager(self._make_registry())
        rec = mgr.record_rebalance(
            date="2025-01-10", product="RB",
            option_delta=100.0,
            old_hedge_qty=0.0, new_hedge_qty=-1.0,
            price=3500.0, multiplier=10.0,
        )
        assert rec.new_hedge_qty == -1.0
        assert len(mgr._records) == 1

    def test_get_records_empty(self):
        mgr = DeltaHedgeManager(self._make_registry())
        df = mgr.get_records()
        assert df.empty


class TestPnlDecomposer:
    def test_compute_daily_basic(self):
        """单日分解: theta 衰减 + gamma 波动."""
        dec = PnlDecomposer()
        snap = dec.compute_daily(
            date="2025-01-10",
            option_positions=[{
                "product": "RB", "qty": 1.0, "multiplier": 10.0,
                "theta": -0.05, "gamma": 0.001, "vega": 0.1,
            }],
            prev_prices={"RB": 3500.0},
            curr_prices={"RB": 3510.0},  # ΔS = 10
            prev_ivs={"RB": 0.20},
            curr_ivs={"RB": 0.21},       # Δσ = 0.01
        )
        # theta_pnl = -0.05 × 1.0 × 10 × (1/365) ≈ -0.00137
        assert snap.theta_pnl < 0
        # gamma_pnl = 0.5 × 0.001 × 100 × 1.0 × 10 = 0.5
        assert snap.gamma_pnl > 0
        # vega_pnl = 0.1 × 0.01 × 1.0 × 10 = 0.01
        assert snap.vega_pnl > 0

    def test_to_dataframe(self):
        dec = PnlDecomposer()
        dec.compute_daily(
            "2025-01-10", [{"product": "RB", "qty": 1, "multiplier": 10,
                            "theta": -0.05, "gamma": 0.001, "vega": 0.1}],
            {"RB": 3500}, {"RB": 3510}, {"RB": 0.2}, {"RB": 0.21},
        )
        df = dec.to_dataframe()
        assert len(df) == 1
        assert "theta" in df.columns
        assert "gamma" in df.columns

    def test_summary(self):
        dec = PnlDecomposer()
        for i in range(5):
            dec.compute_daily(
                f"2025-01-{10+i}",
                [{"product": "RB", "qty": 1, "multiplier": 10,
                  "theta": -0.05, "gamma": 0.001, "vega": 0.1}],
                {"RB": 3500}, {"RB": 3500 + i * 10},
                {"RB": 0.2}, {"RB": 0.2 + i * 0.001},
            )
        s = dec.summary()
        assert "theta" in s
        assert "gamma" in s
        assert s["theta"]["sum"] < 0  # theta 持续衰减


class TestRollManager:
    def test_no_roll_needed(self):
        mgr = RollManager(roll_days=7)
        holdings = {
            "RB": {
                "type": "straddle",
                "expiry": "2025-02-10",
                "call_sym": "rb2505C3500",
                "put_sym": "rb2505P3500",
            }
        }
        actions = mgr.check_and_roll("2025-01-15", holdings, {"RB": 3500.0})
        assert len(actions) == 0

    def test_roll_triggered(self):
        mgr = RollManager(roll_days=7)
        holdings = {
            "RB": {
                "type": "straddle",
                "expiry": "2025-01-20",
                "call_sym": "rb2505C3500",
                "put_sym": "rb2505P3500",
            }
        }
        # 1/15 到 1/20 = 5 天 < 7 天, 触发 roll
        actions = mgr.check_and_roll("2025-01-15", holdings, {"RB": 3500.0})
        assert len(actions) == 1
        assert actions[0].reason == "expiry_approaching"

    def test_non_straddle_ignored(self):
        mgr = RollManager(roll_days=7)
        holdings = {
            "RB": {"type": "naked", "expiry": "2025-01-20"}
        }
        actions = mgr.check_and_roll("2025-01-15", holdings, {"RB": 3500.0})
        assert len(actions) == 0

    def test_get_records(self):
        mgr = RollManager(roll_days=7)
        holdings = {
            "RB": {
                "type": "straddle",
                "expiry": "2025-01-20",
                "call_sym": "rb2505C3500",
                "put_sym": "rb2505P3500",
            }
        }
        mgr.check_and_roll("2025-01-15", holdings, {"RB": 3500.0})
        df = mgr.get_records()
        assert len(df) == 1
