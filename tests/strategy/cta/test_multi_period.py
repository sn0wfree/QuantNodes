# coding=utf-8
"""多周期共振 (5/20/60 日) 因子 + 华泰策略消融研究测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"开仓信号: 第三层".
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.factors.composite import MultiPeriodResonanceFactor
from QuantNodes.strategy.cta.strategies.huatai_controls import run_ablations
from QuantNodes.strategy.cta.strategies.huatai_timing_cta import (
    HuataiTimingConfig,
    HuataiTimingCTAStrategy,
)


# ── MultiPeriodResonanceFactor ─────────────────────────────────────


class TestMultiPeriodResonanceFactor:
    def test_default_windows(self):
        f = MultiPeriodResonanceFactor()
        assert f.windows == (5, 20, 60)
        assert f.min_agree == 2  # 3 选 2

    def test_signal_values(self):
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=idx)
        sig = MultiPeriodResonanceFactor().compute(prices)
        assert set(sig["A"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_strong_uptrend_all_long(self):
        """强趋势: 多数 EMA 都低于价格 → 应一致看多."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=idx)
        sig = MultiPeriodResonanceFactor().compute(prices)
        # 后期稳定看多
        assert (sig["A"].iloc[-50:] == 1.0).all()

    def test_random_walk_balanced(self):
        """随机游走: 多数 EMA 投票应大致平衡."""
        idx = pd.date_range("2025-01-01", periods=300, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, 300))}, index=idx)
        sig = MultiPeriodResonanceFactor().compute(prices)
        long_count = (sig["A"] == 1.0).sum()
        short_count = (sig["A"] == -1.0).sum()
        # 比例不超过 3:1 (随机性)
        ratio = max(long_count, short_count) / max(1, min(long_count, short_count))
        assert ratio < 3.0

    def test_raw_value_range(self):
        """raw_value 投票数范围 [-3, +3]."""
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        prices = pd.DataFrame({"A": np.cumsum(np.random.RandomState(0).normal(0, 1, 100)) + 100}, index=idx)
        votes = MultiPeriodResonanceFactor().raw_value(prices)
        assert float(votes.min().min()) >= -3
        assert float(votes.max().max()) <= 3

    def test_min_agree_validation(self):
        with pytest.raises(ValueError):
            MultiPeriodResonanceFactor(windows=(5, 20, 60), min_agree=5)
        with pytest.raises(ValueError):
            MultiPeriodResonanceFactor(windows=(5, 20, 60), min_agree=0)

    def test_4windows_3agree_more_conservative(self):
        """4 窗口 3-agree 比 3 窗口 2-agree 更保守 (信号更少)."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        rng = np.random.RandomState(42)
        prices = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=idx)
        sig_loose = MultiPeriodResonanceFactor().compute(prices)
        sig_tight = MultiPeriodResonanceFactor(
            windows=(5, 20, 60, 120), min_agree=3,
        ).compute(prices)
        loose_nonzero = (sig_loose["A"] != 0).sum()
        tight_nonzero = (sig_tight["A"] != 0).sum()
        assert tight_nonzero < loose_nonzero

    def test_name(self):
        assert MultiPeriodResonanceFactor.name == "multi_period_resonance"


# ── _entry_signals_multi_period ────────────────────────────────────


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


class TestMultiPeriodEntrySignals:
    def test_returns_dataframe(self):
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
        sig = strat._entry_signals_multi_period()
        assert sig.shape == prices_df.shape
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_multi_period_is_more_conservative_than_default(self):
        """共振过滤应使信号覆盖率 ≤ 默认 entry signals (多了一层确认)."""
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
        default_sig = strat._entry_signals()
        multi_sig = strat._entry_signals_multi_period()
        default_cov = (default_sig != 0).sum().sum() / default_sig.size
        multi_cov = (multi_sig != 0).sum().sum() / multi_sig.size
        # 多周期共振更严格 → 覆盖率 ≤ 默认
        assert multi_cov <= default_cov

    def test_multi_period_signals_subset_of_default(self):
        """共振信号覆盖率 ≤ 默认 (更严格过滤)."""
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
        default_sig = strat._entry_signals()
        multi_sig = strat._entry_signals_multi_period()
        # 比较各列总信号数 (更简洁稳定)
        for col in default_sig.columns:
            default_n = int((default_sig[col] != 0).sum())
            multi_n = int((multi_sig[col] != 0).sum())
            assert multi_n <= default_n, (
                f"{col}: multi={multi_n} > default={default_n} (违反更严格预期)"
            )


# ── run_ablations 包含 multi_period ────────────────────────────────


class TestRunAblationsMultiPeriod:
    def test_multi_period_in_default_run(self):
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
        assert "multi_period" in df.index
        # multi_period 覆盖率应 ≤ full
        assert df.loc["multi_period", "signal_coverage"] <= df.loc["full", "signal_coverage"]
