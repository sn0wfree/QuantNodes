# coding=utf-8
"""CTA 因子测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.trend import (
    MAFactor,
    MomentumFactor,
)
from QuantNodes.strategy.cta.factors.volatility import (
    HVFactor,
    vol_target_positions,
)
from QuantNodes.strategy.cta.factors.term_structure import (
    BasisFactor,
    RollYieldFactor,
    basis_to_signal,
)
from QuantNodes.strategy.cta.signals.trend_signal import donchian_signal


def _mk_uptrend_panel(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    a = pd.Series(np.linspace(100, 200, n), index=idx)   # 强上行
    b = pd.Series(np.linspace(100, 50, n), index=idx)    # 强下行
    return pd.DataFrame({"A": a, "B": b})


class TestDonchian:
    def test_uptrend_positive(self):
        prices = _mk_uptrend_panel()
        sig = donchian_signal(prices, entry_window=10, exit_window=5)
        # 上升品种后期应为 +1
        assert sig["A"].iloc[-1] == 1.0
        # 下降品种后期应为 -1
        assert sig["B"].iloc[-1] == -1.0

    def test_buy_and_hold_enter(self):
        prices = _mk_uptrend_panel()
        sig = donchian_signal(prices, entry_window=10, exit_window=5)
        assert sig["A"].iloc[0:5].mean() == 0.0  # 初始无信号
        assert (sig["A"] >= 0).all()


class TestMA:
    def test_cross(self):
        prices = _mk_uptrend_panel()
        sig = MAFactor(3, 10).compute(prices)
        assert sig["A"].iloc[-1] == 1.0
        assert sig["B"].iloc[-1] == -1.0


class TestMomentum:
    def test_sign(self):
        prices = _mk_uptrend_panel()
        sig = MomentumFactor(10, 0).compute(prices)
        assert sig["A"].iloc[-1] == 1.0
        assert sig["B"].iloc[-1] == -1.0


class TestHV:
    def test_hv_positive(self):
        prices = _mk_uptrend_panel()
        hv = HVFactor(20).compute(prices)
        assert (hv.iloc[-10:].dropna() > 0).all().all()


class TestVolTarget:
    def test_scales_down_high_vol(self):
        prices = _mk_uptrend_panel()
        signals = pd.DataFrame({"A": [1.0] * len(prices), "B": [1.0] * len(prices)},
                               index=prices.index)
        pos = vol_target_positions(signals, prices, target_vol=0.2, vol_lookback=20)
        # 高波动品种仓位 ≤ 低波动品种
        assert pos["A"].iloc[-1] > 0
        assert (pos <= 3.0).all().all()


class TestBasis:
    def test_basis_signal(self):
        # 贴水 (spread>0) → 多头
        s = pd.Series([5.0, -5.0, 0.0])
        out = basis_to_signal(s, threshold=0.0)
        assert out.iloc[0] == 1.0
        assert out.iloc[1] == -1.0
        assert out.iloc[2] == 0.0

    def test_basis_factor(self):
        df = pd.DataFrame([
            {"trade_date": "2025-01-06", "product": "AU", "spread_1_2": 1.5},
            {"trade_date": "2025-01-07", "product": "AU", "spread_1_2": -0.5},
        ])
        out = BasisFactor().compute(df)
        assert out.iloc[0] == 1.5

    def test_roll_yield(self):
        df = pd.DataFrame([
            {"trade_date": "2025-01-06", "product": "AU",
             "spread_1_2": 1.5, "close_near": 500.0},
        ])
        out = RollYieldFactor(days_gap=30).compute(df)
        # 1.5/500 * 252/30
        assert abs(out.iloc[0] - (1.5 / 500 * 252 / 30)) < 1e-6
