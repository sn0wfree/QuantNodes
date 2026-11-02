# coding=utf-8
"""震荡交易 RSI 回落退出 + TR 分位截断测试.

研报: 华泰金工《时序CTA方法论综述》§"实证分析 — 离场信号".
    - 震荡交易逻辑性止损: 空单 RSI 回落至 60 以内平空; 多单 RSI 回升至 40 以上平多
    - 趋势交易 ATR 移动止损: TR 5%-95% 动态分位数截断
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.exits.chandelier import (
    _atr_clipped,
    chandelier_stop_prices,
)
from QuantNodes.strategy.cta.exits.range_exit import (
    range_rsi_exit_check,
    range_rsi_exit_signal,
)
from QuantNodes.strategy.cta.factors.indicators import atr


def _mk_idx(n=100):
    return pd.date_range("2025-01-01", periods=n, freq="B")


class TestRangeRsiExitSignal:
    def test_long_exit_when_rsi_rises_above_40(self):
        """多单: RSI 回升至 40 以上 → 平多."""
        idx = _mk_idx(10)
        position = pd.Series([1.0] * 10, index=idx)
        rsi = pd.Series([25.0, 28.0, 32.0, 35.0, 38.0, 41.0, 45.0, 50.0, 55.0, 60.0], index=idx)
        sig = range_rsi_exit_signal(position, rsi, exit_long=40.0, exit_short=60.0)
        # RSI=41 起触发
        assert not sig.iloc[:5].any()
        assert sig.iloc[5]

    def test_short_exit_when_rsi_falls_below_60(self):
        """空单: RSI 回落至 60 以内 → 平空."""
        idx = _mk_idx(10)
        position = pd.Series([-1.0] * 10, index=idx)
        rsi = pd.Series([75.0, 72.0, 68.0, 65.0, 62.0, 59.0, 55.0, 50.0, 45.0, 40.0], index=idx)
        sig = range_rsi_exit_signal(position, rsi, exit_long=40.0, exit_short=60.0)
        # RSI=59 起触发
        assert not sig.iloc[:5].any()
        assert sig.iloc[5]

    def test_no_exit_when_position_flat(self):
        """空仓 → 永不触发."""
        idx = _mk_idx(10)
        position = pd.Series([0.0] * 10, index=idx)
        rsi = pd.Series([50.0] * 10, index=idx)
        sig = range_rsi_exit_signal(position, rsi)
        assert not sig.any()

    def test_direction_separation(self):
        """多单不会被空单阈值误触发 (RSI=70 不应平多单)."""
        idx = _mk_idx(5)
        position_long = pd.Series([1.0] * 5, index=idx)
        rsi_high = pd.Series([70.0, 72.0, 75.0, 80.0, 85.0], index=idx)
        sig = range_rsi_exit_signal(position_long, rsi_high, exit_long=40.0, exit_short=60.0)
        # 多单的退出条件是 RSI>40 → RSI=70 也会触发 (RSI回升=反转兑现)
        # 这里验证的是: 多单阈值用 exit_long, 不受 exit_short 影响
        assert sig.all()

    def test_index_mismatch_raises(self):
        import pytest
        idx1 = _mk_idx(5)
        idx2 = pd.date_range("2030-01-01", periods=5, freq="B")
        with pytest.raises(ValueError):
            range_rsi_exit_signal(
                pd.Series([1.0] * 5, index=idx1),
                pd.Series([50.0] * 5, index=idx2),
            )

    def test_check_helper(self):
        """单时点判定函数."""
        assert range_rsi_exit_check(1, 45.0)      # 多单 RSI>40
        assert not range_rsi_exit_check(1, 35.0)  # 多单 RSI<40 不平
        assert range_rsi_exit_check(-1, 55.0)     # 空单 RSI<60
        assert not range_rsi_exit_check(-1, 65.0)
        assert not range_rsi_exit_check(0, 50.0)  # 空仓

    def test_custom_thresholds(self):
        """自定义阈值生效."""
        idx = _mk_idx(5)
        position = pd.Series([1.0] * 5, index=idx)
        rsi = pd.Series([30.0, 35.0, 48.0, 52.0, 55.0], index=idx)
        sig = range_rsi_exit_signal(position, rsi, exit_long=45.0)
        assert sig.iloc[2]  # RSI=48 > 45


class TestAtrClipped:
    def _mk_ohlc_with_extreme(self, n=60):
        """构造含单日极端 TR 的 OHLC."""
        idx = _mk_idx(n)
        rng = np.random.RandomState(42)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        ohlc = pd.DataFrame({
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }, index=idx)
        # 注入单日极端波动 (模拟涨跌停)
        ohlc.iloc[30, ohlc.columns.get_loc("high")] = ohlc["close"].iloc[30] + 30.0
        return ohlc

    def test_clip_reduces_extreme_impact(self):
        """极端单日 TR 被截断 → 截断版 ATR < 未截断版 (在冲击日附近)."""
        ohlc = self._mk_ohlc_with_extreme()
        a_raw = atr(ohlc, 14)
        a_clip = _atr_clipped(ohlc, 14)
        # 冲击日之后的窗口内, 截断版应显著更低
        tail = slice(35, 55)
        assert a_clip.iloc[tail].mean() < a_raw.iloc[tail].mean()

    def test_clip_smooth_normal_data(self):
        """正常波动数据: 截断影响小 (截断与未截断接近)."""
        idx = _mk_idx(60)
        rng = np.random.RandomState(7)
        close = 100 + np.cumsum(rng.normal(0, 0.3, 60))
        ohlc = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        }, index=idx)
        a_raw = atr(ohlc, 14).dropna()
        a_clip = _atr_clipped(ohlc, 14).dropna()
        # 正态波动下差异 < 30%
        ratio = (a_clip / a_raw).dropna()
        assert ratio.between(0.7, 1.3).mean() > 0.8

    def test_chandelier_stop_prices_with_clip(self):
        """tr_clip=True 时 stop 价位可正常计算 (含极端TR冲击)."""
        ohlc = self._mk_ohlc_with_extreme()
        stops_clip = chandelier_stop_prices(ohlc, window=22, atr_window=14, mult=3.0, tr_clip=True)
        stops_raw = chandelier_stop_prices(ohlc, window=22, atr_window=14, mult=3.0, tr_clip=False)
        assert list(stops_clip.columns) == ["long_stop", "short_stop"]
        # 截断后 ATR 更小 → 冲击后的 long_stop (高点-ATR) 应高于未截断版 (止损更贴近高点)
        # short_stop (低点+ATR) 应低于未截断版
        tail = slice(40, 55)
        assert (stops_clip["long_stop"].iloc[tail] >= stops_raw["long_stop"].iloc[tail] - 1e-9).all()
        assert (stops_clip["short_stop"].iloc[tail] <= stops_raw["short_stop"].iloc[tail] + 1e-9).all()

    def test_chandelier_clip_off_matches_legacy(self):
        """tr_clip=False 时与旧版行为一致 (用 indicators.atr)."""
        ohlc = self._mk_ohlc_with_extreme()
        stops_off = chandelier_stop_prices(ohlc, window=22, atr_window=14, mult=3.0, tr_clip=False)
        roll_high = ohlc["high"].rolling(22).max()
        roll_low = ohlc["low"].rolling(22).min()
        a = atr(ohlc, 14)
        expected_long = roll_high - 3.0 * a
        expected_short = roll_low + 3.0 * a
        pd.testing.assert_series_equal(stops_off["long_stop"], expected_long, check_names=False)
        pd.testing.assert_series_equal(stops_off["short_stop"], expected_short, check_names=False)
