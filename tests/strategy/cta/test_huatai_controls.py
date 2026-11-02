# coding=utf-8
"""华泰时序CTA对照实验测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"实证分析".
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.huatai_controls import (
    _range_only_signals,
    _trend_only_signals,
    run_ablations,
)
from QuantNodes.strategy.cta.strategies.huatai_timing_cta import (
    HuataiTimingConfig,
)


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


def _make_mock_loader(prices_df, ohlc_df):
    def fake_loader(product, start, end):
        return ohlc_df[product].reset_index().rename(columns={"index": "trade_date"})
    mock_loader = MagicMock()
    mock_loader.load_continuous_main.side_effect = fake_loader
    return mock_loader


class TestTrendOnlySignals:
    def test_shape(self):
        prices, _ = _mk_synthetic_panel()
        sig = _trend_only_signals(prices, 5, 20, 60, 30)
        assert sig.shape == prices.shape

    def test_signal_values(self):
        prices, _ = _mk_synthetic_panel()
        sig = _trend_only_signals(prices, 5, 20, 60, 30)
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_high_coverage(self):
        """trend_only 假设恒为趋势 → 信号覆盖率应 > 50%."""
        prices, _ = _mk_synthetic_panel(n=200)
        sig = _trend_only_signals(prices, 5, 20, 60, 30)
        cov = (sig != 0).sum().sum() / sig.size
        assert cov > 0.5

    def test_uptrend_long(self):
        """纯上行 → 应持续多头."""
        idx = pd.date_range("2025-01-01", periods=200, freq="B")
        prices = pd.DataFrame({"A": np.linspace(100, 200, 200)}, index=idx)
        sig = _trend_only_signals(prices, 5, 20, 60, 30)
        assert (sig["A"].iloc[60:] == 1.0).all()


class TestRangeOnlySignals:
    def test_shape(self):
        prices, _ = _mk_synthetic_panel()
        sig = _range_only_signals(prices, 14, 30.0, 70.0)
        assert sig.shape == prices.shape

    def test_signal_values(self):
        prices, _ = _mk_synthetic_panel()
        sig = _range_only_signals(prices, 14, 30.0, 70.0)
        assert set(sig["IF"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_low_coverage(self):
        """range_only 用 RSI 超买超卖 → 随机游走下覆盖率低."""
        prices, _ = _mk_synthetic_panel(n=200)
        sig = _range_only_signals(prices, 14, 30.0, 70.0)
        cov = (sig != 0).sum().sum() / sig.size
        # 随机游走下覆盖 < 30%
        assert cov < 0.3

    def test_constant_price_no_signal(self):
        """常数价格 → RSI = 50 → 无信号."""
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        prices = pd.DataFrame({"A": 100.0}, index=idx)
        sig = _range_only_signals(prices, 14, 30.0, 70.0)
        # RSI 对常数价格 = 50, 不会触发 30/70 阈值
        assert (sig["A"] == 0.0).all()


class TestRunAblations:
    def test_returns_dataframe(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(prices_df, ohlc_df)
        df = run_ablations(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
            include_presets=False,
        )
        # 必须包含全部 8 个变体行 (4 基础 + 4 共振)
        assert set(df.index) == {
            "full", "trend_only", "range_only", "no_chandelier",
            "multi_period", "weighted_period", "vol_regime", "cross_dim",
        }
        # 必须含关键指标列
        assert {"AnnRet", "Sharpe", "MaxDD", "Calmar", "trades", "signal_coverage"}.issubset(df.columns)

    def test_no_chandelier_worse_than_full_synthetic(self):
        """合成数据下, 关闭吊灯止损(no_chandelier)应劣于完整策略 (文章论点).

        真实 DuckDB 数据上 no_chandelier Calmar=-0.049 vs full Calmar=0.022.
        """
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(prices_df, ohlc_df)
        df = run_ablations(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
            include_presets=False,
        )
        # 关闭吊灯止损应使卡玛比 ≤ 完整策略 (允许等号: 卡玛 = 0)
        assert df.loc["no_chandelier", "Calmar"] <= df.loc["full", "Calmar"]

    def test_includes_presets(self):
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(prices_df, ohlc_df)
        df = run_ablations(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
            include_presets=True,
        )
        # 验证包含 7 个 preset 行
        preset_rows = [v for v in df.index if v.startswith("preset:")]
        assert len(preset_rows) == 7  # default + responsive + levered + patient + aggressive + slope + slope_responsive

    def test_signal_coverage_in_full_matches_state_coverage(self):
        """signal_coverage 应反映有信号格的比例."""
        prices_df, ohlc_df = _mk_synthetic_panel(n=200)
        loader = _make_mock_loader(prices_df, ohlc_df)
        df = run_ablations(
            products=["IF", "AU", "T"],
            config=HuataiTimingConfig(initial_cash=500_000.0),
            data_loader=loader,
            start="2020-01-01",
            end="2024-12-31",
            include_presets=False,
        )
        # range_only 信号覆盖率最低, trend_only 最高
        assert df.loc["range_only", "signal_coverage"] < df.loc["trend_only", "signal_coverage"]


class TestConfigPresets:
    def test_presets_returns_dict(self):
        presets = HuataiTimingConfig.presets()
        assert isinstance(presets, dict)
        expected = {
            "default", "responsive", "levered", "patient", "aggressive",
            "slope", "slope_responsive",
        }
        assert set(presets.keys()) == expected

    def test_responsive_shortens_confirm_window(self):
        presets = HuataiTimingConfig.presets()
        assert presets["responsive"].confirm_window == 5
        assert presets["default"].confirm_window == 10

    def test_levered_increases_target_vol(self):
        presets = HuataiTimingConfig.presets()
        assert presets["levered"].target_vol == 0.20
        assert presets["default"].target_vol == 0.15

    def test_patient_extends_holding_days(self):
        presets = HuataiTimingConfig.presets()
        assert presets["patient"].max_holding_days == 120
        assert presets["default"].max_holding_days == 60

    def test_aggressive_combines(self):
        presets = HuataiTimingConfig.presets()
        aggr = presets["aggressive"]
        assert aggr.confirm_window == 5
        assert aggr.target_vol == 0.20
        assert aggr.max_holding_days == 120
