# coding=utf-8
"""基差动量/持仓背离/风险类因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.basis_momentum import (
    BasisLevelFactor,
    BasisMomentumFactor,
)
from QuantNodes.strategy.cta.factors.volume import OIPriceDivergenceFactor
from QuantNodes.strategy.cta.factors.risk import (
    DownsideVolFactor,
    MaxDrawdownFactor,
    TailRiskFactor,
)


def _mk_basis(n=120):
    """基差 long-form: 基差率先升后降."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rate = np.concatenate([np.linspace(-0.01, 0.02, n // 2),
                           np.linspace(0.02, -0.01, n - n // 2)])
    return pd.DataFrame({
        "trade_date": idx, "symbol": "au", "near_basis_rate": rate,
        "dom_basis_rate": rate * 0.8,
    })


def _mk_prices(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    b = np.linspace(100, 200, n)
    c = np.linspace(200, 100, n)
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def _mk_oi(prices):
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {c: rng.randint(5000, 20000, len(prices)) for c in prices.columns},
        index=prices.index,
    )


class TestBasisMomentum:
    def test_panel_shape(self):
        basis = _mk_basis()
        sig = BasisMomentumFactor(5).compute(basis)
        assert sig.shape[0] == len(basis["trade_date"].unique())
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_rising_basis_long(self):
        # 基差率持续上升段 → 多头
        basis = _mk_basis()
        sig = BasisMomentumFactor(5).compute(basis)
        # 前半段基差率上升
        assert (sig["AU"].iloc[20:40] > 0).any()

    def test_dom_basis_col(self):
        basis = _mk_basis()
        sig = BasisMomentumFactor(5, value_col="dom_basis_rate").compute(basis)
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})


class TestBasisLevel:
    def test_high_positive_long(self):
        # 基差率恒正 → 现货升水 → 多头
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        basis = pd.DataFrame({
            "trade_date": idx, "symbol": "au", "near_basis_rate": 0.02,
        })
        sig = BasisLevelFactor(5, threshold=0.005).compute(basis)
        assert (sig["AU"].iloc[5:] == 1.0).all()  # 前 5 天 rolling 未满

    def test_negative_short(self):
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        basis = pd.DataFrame({
            "trade_date": idx, "symbol": "au", "near_basis_rate": -0.02,
        })
        sig = BasisLevelFactor(5, threshold=0.005).compute(basis)
        assert (sig["AU"].iloc[5:] == -1.0).all()


class TestOIPriceDivergence:
    def test_up_oi_up_long(self):
        # 价涨 + 仓增 → 顺势多
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
        oi = pd.DataFrame({"A": np.linspace(5000, 10000, n)}, index=idx)
        sig = OIPriceDivergenceFactor(5, 5).compute(prices, oi)
        assert sig["A"].iloc[-1] == 1.0

    def test_up_oi_down_short(self):
        # 价涨 + 仓减 → 顶背离 → 空
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
        oi = pd.DataFrame({"A": np.linspace(10000, 5000, n)}, index=idx)
        sig = OIPriceDivergenceFactor(5, 5).compute(prices, oi)
        assert sig["A"].iloc[-1] == -1.0

    def test_values(self):
        prices = _mk_prices()
        oi = _mk_oi(prices)
        sig = OIPriceDivergenceFactor(5, 5).compute(prices, oi)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})


class TestRisk:
    def test_downside_vol(self):
        prices = _mk_prices()
        sig = DownsideVolFactor(20, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_max_drawdown(self):
        prices = _mk_prices()
        sig = MaxDrawdownFactor(60, 1.0).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_tail_risk(self):
        prices = _mk_prices()
        sig = TailRiskFactor(60, 0.05).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_downside_vol_constant(self):
        # 单调上行 (无负收益) → 下行波动 0 → 无信号
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
        sig = DownsideVolFactor(20, 1.0).compute(prices)
        assert (sig["A"] == 0).all()

    def test_drawdown_triggers(self):
        # 冲高回落 → 深回撤 → 信号
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        ramp = np.linspace(100, 300, 150)
        crash = np.linspace(300, 150, 50)
        prices = pd.DataFrame({"A": np.concatenate([ramp, crash])}, index=idx)
        sig = MaxDrawdownFactor(60, 1.0).compute(prices)
        assert (sig["A"] != 0).any()
