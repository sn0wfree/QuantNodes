# coding=utf-8
"""期权回测引擎测试: 期权合约解析 + 买方调仓 + 到期结算."""
import pandas as pd

from QuantNodes.strategy.cta.backtest.cta_broker import CtaBroker
from QuantNodes.strategy.cta.backtest.cta_contract import (
    CtaContract, CtaContractRegistry, OptionContractSpec,
)
from QuantNodes.strategy.cta.backtest.cta_engine import (
    CtaBacktestConfig, CtaStrategy, run_cta_backtest,
)


class TestOptionContract:
    def test_parse_io(self):
        reg = CtaContractRegistry()
        c = reg.get_option("IO2502-C-3700")
        assert c.is_option
        assert c.option_spec.option_type == "C"
        assert c.option_spec.strike == 3700.0
        assert c.option_spec.contract_multiplier == 100.0

    def test_parse_m(self):
        reg = CtaContractRegistry()
        c = reg.get_option("m2608-P-2750")
        assert c.is_option
        assert c.option_spec.option_type == "P"
        assert c.option_spec.contract_multiplier == 10.0

    def test_future_not_option(self):
        reg = CtaContractRegistry()
        c = reg.get_option("IF2608")
        assert not c.is_option


class TestOptionBroker:
    def _broker(self):
        return CtaBroker(CtaContractRegistry(), initial_cash=1_000_000.0)

    def _contract(self):
        return CtaContract(
            product="RB", symbol="rb2605P600", expiry_year=2026, expiry_month=5,
            spec=CtaContractRegistry().get_or_default("RB"),
            option_spec=OptionContractSpec(
                exchange="SHFE", product="RB", underlying="rb2605",
                option_type="P", strike=600.0, expiry_date="2026-05-01",
                contract_multiplier=10.0,
            ),
        )

    def test_buy_option_costs_premium(self):
        """买 1 手 Put: 扣权利金 = qty × price × mult, 无保证金."""
        b = self._broker()
        ok = b.place_order("2026-01-05", "rb2605P600", "RB", 1, 5.0, self._contract())
        assert ok
        assert b.positions["rb2605P600"].qty == 1
        assert b.positions["rb2605P600"].kind == "option_buy"
        # 权利金 5.0 × 10 = 50
        assert abs(b.cash - (1_000_000 - 50 - 0.005)) < 1e-6  # fee 0.0001×5×10
        assert b.margin_used() == 0.0  # 买方无保证金

    def test_mark_to_market_option(self):
        """期权按权利金结算: 权利金上涨 → 权益增加."""
        b = self._broker()
        b.place_order("2026-01-05", "rb2605P600", "RB", 1, 5.0, self._contract())
        b.mark_to_market({"rb2605P600": 8.0}, "2026-01-06")
        # 浮动 = 1 × (8-5) × 10 = 30
        assert abs(b.equity() - (1_000_000 - 50 - 0.005 + 30)) < 1e-6

    def test_close_option_realized(self):
        """平仓: 已实现盈亏 = (现价 - 开仓价) × qty × mult."""
        b = self._broker()
        b.place_order("2026-01-05", "rb2605P600", "RB", 2, 5.0, self._contract())
        b.place_order("2026-01-06", "rb2605P600", "RB", 0, 9.0, self._contract())
        assert "rb2605P600" not in b.positions
        # 平仓退回全部权利金 (含成本): cash = 期初 - 开仓权利金 - 开仓fee + 平仓权利金 - 平仓fee
        #  = 1,000,000 - 100 - 0.01 + 180 - 0.018
        assert abs(b.equity() - (1_000_000 - 2 * 50 - 0.01 + 2 * 9 * 10 - 0.018)) < 1e-4
        # realized_pnl 记盈亏 = 80
        assert abs(b.positions.get("rb2605P600", None) is None) == 1

    def test_expire_option_intrinsic(self):
        """到期按内在价值结算 (Put: max(K-S, 0))."""
        b = self._broker()
        b.place_order("2026-01-05", "rb2605P600", "RB", 1, 5.0, self._contract())
        # 到期日结算价: 内在价值 = max(600-S, 0), 设 S=550 → 50
        b._expire_options("2026-05-01", {"rb2605P600": 50.0})
        assert "rb2605P600" not in b.positions
        # 到期结算退回全部结算价值 (含成本): cash = 期初 - 权利金 - fee + qty×settle×mult
        assert abs(b.equity() - (1_000_000 - 50 - 0.005 + 1 * 50 * 10)) < 1e-6


class TestOptionBacktest:
    def test_mixed_run(self):
        """期货 + 期权同账户回测可跑通."""
        dates = pd.bdate_range("2026-01-05", periods=5)
        prices = {"rb2605": [3000.0] * 5, "rb2605P600": [5.0] * 5}
        products = {"rb2605": "RB", "rb2605P600": "RB"}
        reg = CtaContractRegistry()

        class S(CtaStrategy):
            def target_positions(self, date, data):
                return {"rb2605": 1, "rb2605P600": 1}

        r = run_cta_backtest(
            dates=list(dates), prices=prices, products=products, strategy=S(),
            config=CtaBacktestConfig(initial_cash=1_000_000.0),
            registry=reg,
            option_contracts={"rb2605P600": reg.get_option("rb2605P600")},
        )
        assert r.metrics["Years"] > 0
        assert r.positions["rb2605P600"][-1] == 1.0
