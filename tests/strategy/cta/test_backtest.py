# coding=utf-8
"""CTA 回测引擎测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.backtest.cta_broker import CtaBroker
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_engine import (
    CtaBacktestConfig,
    CtaStrategy,
    run_cta_backtest,
)
from QuantNodes.strategy.cta.backtest.cta_risk import (
    CompositeRisk,
    MarginRisk,
    PositionLimitRisk,
)


def _make_registry():
    """构造带 AU/RB 规格的内存注册表."""
    reg = CtaContractRegistry.__new__(CtaContractRegistry)
    from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractSpec
    reg._specs = {
        "AU": CtaContractSpec("SHFE", "AU", 1000.0, "元/克", 0.02, 0.15),
        "RB": CtaContractSpec("SHFE", "RB", 10.0, "元/吨", 1.0, 0.12),
    }
    reg._margin = {}
    return reg


class _ConstStrategy(CtaStrategy):
    """固定持仓策略 (调试用)."""

    def __init__(self, targets: dict[str, float]):
        self._targets = targets

    def target_positions(self, date, data):
        return self._targets


class TestCtaBroker:
    def test_open_position_uses_margin(self):
        reg = _make_registry()
        broker = CtaBroker(reg, initial_cash=1_000_000)
        # AU 价格 500, 乘数 1000, 保证金 15% → 每手保证金 = 500*1000*0.15 = 75000
        ok = broker.place_order("2025-01-06", "au2502", "AU", target_qty=1, price=500.0)
        assert ok
        assert broker.positions["au2502"].qty == 1
        # 保证金 75000 + 手续费 50 (500×1000×0.0001)
        assert abs(broker.cash - (1_000_000 - 75000 - 50)) < 1e-6

    def test_margin_requirement(self):
        reg = _make_registry()
        broker = CtaBroker(reg, initial_cash=1_000_000, margin_call=True)
        ok = broker.place_order("2025-01-06", "au2502", "AU", target_qty=1, price=500.0)
        assert ok
        # 保证金 75000, 现金 925000 → 可再开 12 手 (925000/75000≈12.3)
        ok = broker.place_order("2025-01-06", "au2502", "AU", target_qty=13, price=500.0)
        assert not ok  # 保证金不足拒绝

    def test_close_position_releases_cash(self):
        reg = _make_registry()
        broker = CtaBroker(reg, initial_cash=1_000_000)
        broker.place_order("2025-01-06", "au2502", "AU", target_qty=2, price=500.0)
        cash_after_open = broker.cash  # 849900
        # 平 1 手: 释放 75000 保证金 + 盈亏 (550-500)*1*1000 - 手续费 55
        broker.place_order("2025-01-07", "au2502", "AU", target_qty=1, price=550.0)
        assert broker.positions["au2502"].qty == 1
        # 平仓净增 = 75000(保证金) + 50000(盈利) - 55(费) = 124945
        assert abs(broker.cash - (cash_after_open + 124945)) < 1e-6

    def test_short_position(self):
        reg = _make_registry()
        broker = CtaBroker(reg, initial_cash=1_000_000, allow_short=True)
        ok = broker.place_order("2025-01-06", "rb2605", "RB", target_qty=-10, price=3200.0)
        assert ok
        assert broker.positions["rb2605"].qty == -10

    def test_reverse_position(self):
        reg = _make_registry()
        broker = CtaBroker(reg, initial_cash=1_000_000)
        broker.place_order("2025-01-06", "au2502", "AU", target_qty=2, price=500.0)
        # 反手 → 空 1
        broker.place_order("2025-01-07", "au2502", "AU", target_qty=-1, price=500.0)
        assert broker.positions["au2502"].qty == -1


class TestRunBacktest:
    def _mk_data(self):
        dates = pd.date_range("2025-01-02", periods=20, freq="B")
        # AU 单调上涨 (多头盈利)
        au = np.linspace(500, 550, 20)
        return dates, {"au2502": au.tolist()}, {"au2502": "AU"}

    def test_simple_long_profits(self):
        dates, prices, products = self._mk_data()
        strat = _ConstStrategy({"au2502": 2.0})
        result = run_cta_backtest(
            dates=list(dates), prices=prices, products=products,
            strategy=strat, config=CtaBacktestConfig(initial_cash=1_000_000),
        )
        assert len(result.nav) == 20
        assert result.nav.iloc[-1] > 1.0  # 上涨行情多头盈利
        assert result.metrics["AnnRet"] > 0

    def test_flat_no_trade(self):
        dates, prices, products = self._mk_data()
        strat = _ConstStrategy({"au2502": 0.0})
        result = run_cta_backtest(
            dates=list(dates), prices=prices, products=products,
            strategy=strat, config=CtaBacktestConfig(initial_cash=1_000_000),
        )
        assert abs(result.nav.iloc[-1] - 1.0) < 1e-6

    def test_with_margin_risk(self):
        dates, prices, products = self._mk_data()
        strat = _ConstStrategy({"au2502": 100.0})  # 巨大持仓
        reg = _make_registry()
        risk = MarginRisk(reg, products, equity=1_000_000, max_margin_ratio=0.8)
        result = run_cta_backtest(
            dates=list(dates), prices=prices, products=products,
            strategy=strat, config=CtaBacktestConfig(initial_cash=1_000_000),
            registry=reg, risk=risk,
        )
        assert len(result.trades) > 0

    def test_position_limit(self):
        dates, prices, products = self._mk_data()
        strat = _ConstStrategy({"au2502": 5.0})
        reg = _make_registry()
        risk = PositionLimitRisk(max_qty_per_product={"AU": 3}, products=products, registry=reg)
        result = run_cta_backtest(
            dates=list(dates), prices=prices, products=products,
            strategy=strat, config=CtaBacktestConfig(initial_cash=1_000_000),
            registry=reg, risk=risk,
        )
        # 持仓被限制为 3 手
        assert result.positions["au2502"][-1] == 3.0


class TestCompositeRisk:
    def test_chained(self):
        reg = _make_registry()
        products = {"au2502": "AU"}
        r1 = PositionLimitRisk(max_qty_per_product={"AU": 3}, products=products, registry=reg)
        r2 = MarginRisk(reg, products, equity=1_000_000, max_margin_ratio=0.8)
        comp = CompositeRisk([r1, r2])
        targets = {"au2502": 5.0}
        mark = {"au2502": 500.0}
        out = comp.apply(targets, mark, 0, [], {})
        assert out["au2502"] <= 3.0  # 先被 r1 限到 3, 再被 r2 缩放
