# coding=utf-8
"""TSI (True Strength Index) 振荡器因子测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.factors.oscillators import TSIFactor


def _mk_uptrend(n=300, start=100.0, end=200.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.linspace(start, end, n)}, index=idx)


def _mk_random_walk(n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)


class TestTSI:
    def test_shape(self):
        prices = _mk_uptrend()
        sig = TSIFactor().compute(prices)
        assert sig.shape == prices.shape
        assert sig.index.equals(prices.index)

    def test_signal_values(self):
        prices = _mk_uptrend()
        sig = TSIFactor().compute(prices)
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_uptrend_positive(self):
        # 单调上行 → TSI 后期持续 > 25 → 多
        sig = TSIFactor(25, 13, 25, -25).compute(_mk_uptrend())
        assert (sig["A"] == 1.0).any()

    def test_downtrend_negative(self):
        # 单调下行 → TSI 后期持续 < -25 → 空
        prices = _mk_uptrend(start=200.0, end=100.0)
        sig = TSIFactor(25, 13, 25, -25).compute(prices)
        assert (sig["A"] == -1.0).any()

    def test_random_walk_neutral(self):
        # 随机游走 → TSI 应在 [-25, 25] 区间振荡 → 信号稀疏
        sig = TSIFactor(25, 13, 25, -25).compute(_mk_random_walk())
        nonzero_ratio = (sig["A"] != 0).mean()
        assert nonzero_ratio < 0.3  # 信号不应超过 30% 时间

    def test_raw_value_range(self):
        # raw_value 应在 [-100, +100] 区间内 (浮点容差 1e-9)
        raw = TSIFactor().raw_value(_mk_uptrend())
        valid = raw["A"].dropna()
        assert valid.min() >= -100 - 1e-9
        assert valid.max() <= 100 + 1e-9

    def test_constant_price(self):
        # 常数价格 → |Δ|=0 → TSI=NaN → compute 不触发阈值 → 全 0
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        prices = pd.DataFrame({"A": 100.0}, index=idx)
        sig = TSIFactor().compute(prices)
        assert (sig["A"] == 0.0).all()

    def test_short_window_more_sensitive(self):
        # 短窗口 r=5 应比长窗口 r=25 触发更多信号
        rw = _mk_random_walk()
        sig_short = TSIFactor(5, 3, 25, -25).compute(rw)
        sig_long = TSIFactor(25, 13, 25, -25).compute(rw)
        assert (sig_short["A"] != 0).sum() >= (sig_long["A"] != 0).sum()

    def test_name(self):
        assert TSIFactor.name == "tsi"
