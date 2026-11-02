# coding=utf-8
"""共振因子变体测试 (加权/波动率体制/跨维度).

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"开仓信号: 第三层".
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.factors.composite import (
    CrossDimensionalResonanceFactor,
    MultiPeriodResonanceFactor,
    VolatilityRegimeResonanceFactor,
)
from QuantNodes.strategy.cta.strategies.huatai_controls import run_ablations
from QuantNodes.strategy.cta.strategies.huatai_timing_cta import (
    HuataiTimingConfig,
    HuataiTimingCTAStrategy,
)


# ── 加权共振 ──────────────────────────────────────────────────────────────────────────


class TestWeightedResonance:
    def test_equal_weights_match_min_agree(self):
        """等权 + min_agree=2 ≡ 2-of-3 多数决."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({
            "A": 100 + np.cumsum(rng.normal(0, 1, 200)),
        }, index=idx)
        eq = MultiPeriodResonanceFactor(windows=(5, 20, 60)).compute(prices)
        w_eq = MultiPeriodResonanceFactor(
            windows=(5, 20, 60), weights=(1, 1, 1),
        ).compute(prices)
        pd.testing.assert_frame_equal(eq, w_eq)

    def test_weighted_recent_heavier(self):
        """近期权重高 → 比等权更敏感 (更多信号)."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({
            "A": 100 + np.cumsum(rng.normal(0, 1, 200)),
        }, index=idx)
        eq = MultiPeriodResonanceFactor(windows=(5, 20, 60))
        w_recent = MultiPeriodResonanceFactor(
            windows=(5, 20, 60), weights=(0.6, 0.3, 0.1),
        )
        n_eq = (eq.compute(prices)["A"] != 0).sum()
        n_w = (w_recent.compute(prices)["A"] != 0).sum()
        # 近期高权重 → 单个 5d 即可触发 (权重 0.6 >= 0.5 × total=1.0)
        # 应比 2-of-3 多数决更宽松
        assert n_w > n_eq

    def test_weights_validation(self):
        with pytest.raises(ValueError):
            MultiPeriodResonanceFactor(windows=(5, 20, 60), weights=(0.5, 0.5))
        with pytest.raises(ValueError):
            MultiPeriodResonanceFactor(windows=(5, 20, 60), weights=(-0.5, 0.5, 0.5))


# ── 波动率体制共振 ──────────────────────────────────────────────────────────────────


class TestVolatilityRegimeResonance:
    def test_returns_zero_in_low_vol(self):
        """低波动率环境下应输出 0 (共振关闭)."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(0)
        close = 100 + np.cumsum(rng.normal(0, 0.001, 200))
        prices = pd.DataFrame({"A": close}, index=idx)
        sig = VolatilityRegimeResonanceFactor(percentile=0.75).compute(prices)
        # 后期 (滚动百分位已建立) 信号应较少
        late = sig["A"].iloc[100:]
        assert (late == 0).sum() > (late != 0).sum()

    def test_signal_values(self):
        """输出值仍为 {-1, 0, +1}."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)
        sig = VolatilityRegimeResonanceFactor().compute(prices)
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_higher_percentile_more_selective(self):
        """更高百分位阈值 → 信号更稀疏."""
        idx = pd.date_range("2025-01-01", periods=300, freq="B")
        rng = np.random.RandomState(42)
        ret = rng.normal(0, 0.01, 300)
        ret[150:] *= 2  # 后半段波动率上升
        prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(ret))}, index=idx)
        sig_loose = VolatilityRegimeResonanceFactor(percentile=0.50).compute(prices)
        sig_tight = VolatilityRegimeResonanceFactor(percentile=0.90).compute(prices)
        loose_n = (sig_loose["A"] != 0).sum()
        tight_n = (sig_tight["A"] != 0).sum()
        assert tight_n <= loose_n

    def test_name(self):
        assert VolatilityRegimeResonanceFactor.name == "vol_regime_resonance"


# ── 跨维度共振 ──────────────────────────────────────────────────────────────────────


class TestCrossDimensionalResonance:
    def test_three_dimensions_vote(self):
        """三维投票: 至少 min_agree_dim 个维度同向 → 信号."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)
        sig = CrossDimensionalResonanceFactor().compute(prices)
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_strong_uptrend_all_three_align(self):
        """强趋势: 三维度应一致 → 信号密集."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=idx)
        sig = CrossDimensionalResonanceFactor().compute(prices)
        # 后期 (数据足够) 应一致看多
        assert (sig["A"].iloc[-50:] == 1.0).all()

    def test_name(self):
        assert CrossDimensionalResonanceFactor.name == "cross_dim_resonance"


# ── 策略入口变体 ────────────────────────────────────────────────────────────────────


def _mk_synthetic_panel(n=200, products=("IF", "AU", "T"), seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices, ohlcs = {}, {}
    for product in products:
        drift = {"IF": 0.0008, "AU": 0.0005, "T": 0.0002}[product]
        vol = {"IF": 0.015, "AU": 0.012, "T": 0.003}[product]
        ret = rng.normal(drift, vol, n)
        close = 100 * np.exp(np.cumsum(ret))
        ohlcs[product] = pd.DataFrame({
            "open": close + rng.normal(0, 0.2, n),
            "high": close + rng.uniform(0.5, 1.5, n),
            "low": close - rng.uniform(0.5, 1.5, n),
            "close": close,
        }, index=idx)
        prices[product] = pd.Series(close, index=idx)
    return pd.DataFrame(prices).sort_index(), pd.concat(ohlcs, axis=1).sort_index()


def _make_mock_loader(ohlc_df):
    def fake_loader(product, start, end):
        return ohlc_df[product].reset_index().rename(columns={"index": "trade_date"})
    mock_loader = MagicMock()
    mock_loader.load_continuous_main.side_effect = fake_loader
    return mock_loader


class TestEntrySignalVariants:
    def test_weighted_returns_valid_signal(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(ohlc_df)
        strat = HuataiTimingCTAStrategy(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
        )
        strat.load_panel()
        strat._estimate_states()
        sig = strat._entry_signals_multi_period(
            windows=(5, 20, 60), weights=(0.5, 0.3, 0.2),
        )
        assert sig.shape == prices_df.shape
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_vol_regime_returns_valid_signal(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(ohlc_df)
        strat = HuataiTimingCTAStrategy(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
        )
        strat.load_panel()
        strat._estimate_states()
        sig = strat._entry_signals_vol_regime(percentile=0.75)
        assert sig.shape == prices_df.shape
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_cross_dim_returns_valid_signal(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(ohlc_df)
        strat = HuataiTimingCTAStrategy(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
        )
        strat.load_panel()
        strat._estimate_states()
        sig = strat._entry_signals_cross_dim()
        assert sig.shape == prices_df.shape
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})


class TestRunAblationsIncludesAllVariants:
    def test_all_three_new_variants_present(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(ohlc_df)
        df = run_ablations(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
            include_presets=False,
        )
        expected = {
            "full", "trend_only", "range_only", "no_chandelier",
            "multi_period", "weighted_period", "vol_regime", "cross_dim",
        }
        assert set(df.index) == expected
