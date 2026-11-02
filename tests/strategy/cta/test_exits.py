# coding=utf-8
"""退出机制子模块测试.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.exits import (
    DrawdownCircuitBreaker,
    ExitLayers,
    chandelier_exit_signal,
    chandelier_stop_prices,
    combine_exit_signals,
    evaluate_exit,
    exit_reason,
    holding_days,
    logical_exit_check,
    logical_exit_for_trend,
    regime_change_signals,
    rolling_drawdown,
    stress_test_extreme_days,
    target_vol_leverage,
    time_exit_signals,
)
from QuantNodes.strategy.cta.regime import (
    STATE_RANGE,
    STATE_TREND_UP,
)


def _mk_ohlc(n=200, start=100.0):
    """构造平稳 OHLC (横盘), 用于触发吊灯止损测试."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    close = start + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.uniform(0.5, 1.5, n),
        "low": close - rng.uniform(0.5, 1.5, n),
        "close": close,
    }, index=idx)


def _mk_breakout_then_drop(n=200, start=100.0):
    """构造先上行后暴跌序列 (首日跳空 -20): 多头持仓被吊灯止损触发."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    half = n // 2
    rise = np.linspace(start, start + 30, half)
    # 暴跌首日跳空 -20 点, 之后继续下行
    crash = np.concatenate([[start + 10], np.linspace(start + 10, start - 10, n - half - 1)])
    close = np.concatenate([rise, crash])
    return pd.DataFrame({
        "open": close,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
    }, index=idx)


class TestChandelier:
    def test_stop_prices_shape(self):
        ohlc = _mk_ohlc()
        stops = chandelier_stop_prices(ohlc)
        assert list(stops.columns) == ["long_stop", "short_stop"]
        assert stops.index.equals(ohlc.index)

    def test_stop_prices_long_above_short(self):
        # 多头止损应低于空头止损 (横盘时)
        ohlc = _mk_ohlc()
        stops = chandelier_stop_prices(ohlc)
        valid = stops.dropna()
        assert (valid["long_stop"] < valid["short_stop"]).all()

    def test_exit_signal_long_drop(self):
        # 长多后跳空暴跌 → 应触发多头止损
        ohlc = _mk_breakout_then_drop()
        position = pd.Series(1, index=ohlc.index)
        # 紧窗口 + 大倍数才能在非剧烈行情触发
        sig = chandelier_exit_signal(ohlc, position, window=5, atr_window=5, mult=3.0)
        assert sig.any()

    def test_exit_signal_no_position(self):
        # 空仓时永远不触发
        ohlc = _mk_breakout_then_drop()
        position = pd.Series(0, index=ohlc.index)
        sig = chandelier_exit_signal(ohlc, position)
        assert not sig.any()


class TestLogicalExit:
    def test_regime_change_detected(self):
        # 状态翻转 (50 位置) → 在 confirm_window=3 处确认
        states = pd.DataFrame({
            "A": ["trend_up"] * 50 + ["range"] * 50,
        }, index=pd.date_range("2025-01-01", periods=100, freq="B"))
        sig = regime_change_signals(states, confirm_window=3)
        # t=52 处 state[52]=range vs state[49]=trend_up → True
        assert bool(sig.iloc[52, 0])
        # 后续 t=53 起 state[t]=state[t-3]=range → False
        assert not bool(sig.iloc[53, 0])

    def test_regime_change_short_ignored(self):
        # 状态翻转 (50 位置) 后又恢复 → 翻转未被确认
        states = pd.DataFrame({
            "A": ["trend_up"] * 50 + ["range"] * 2 + ["trend_up"] * 48,
        }, index=pd.date_range("2025-01-01", periods=100, freq="B"))
        sig = regime_change_signals(states, confirm_window=5)
        # 末尾是 trend_up 持续, 无确认信号
        assert not sig.iloc[-10:].any().any()

    def test_logical_exit_for_trend_long(self):
        # 持多 + 状态从 trend_up 变为 range (index 5) → confirm_window=2 后应触发
        idx = pd.date_range("2025-01-01", periods=10, freq="B")
        states = pd.DataFrame({"A": ["trend_up"] * 5 + ["range"] * 5}, index=idx)
        position = pd.Series([1] * 10, index=idx)
        sig = logical_exit_for_trend(states, position, confirm_window=2)
        # t=6: state[6]=range, state[4]=trend_up → 应触发
        assert bool(sig.iloc[6])
        # t=8: state[8]=range, state[6]=range → 不再触发
        assert not bool(sig.iloc[8])

    def test_logical_exit_check_helper(self):
        assert logical_exit_check(STATE_TREND_UP, +1, STATE_RANGE)
        assert not logical_exit_check(STATE_TREND_UP, +1, STATE_TREND_UP)
        assert not logical_exit_check("range", 0, STATE_TREND_UP)


class TestTimeExit:
    def test_holding_days(self):
        pos = pd.Series([0, 1, 1, 1, 0, 1, 1], index=pd.date_range("2025-01-01", periods=7))
        days = holding_days(pos)
        assert days.iloc[0] == 0
        assert days.iloc[1] == 1
        assert days.iloc[3] == 3
        assert days.iloc[4] == 0  # 平仓
        assert days.iloc[5] == 1
        assert days.iloc[6] == 2

    def test_time_exit_triggered(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="B")
        pos = pd.Series([1] * 10, index=idx)
        pnl = pd.Series([-0.01] * 10, index=idx)
        sig = time_exit_signals(pos, pnl, max_holding_days=5)
        # 第 5 日起 (天数=5) 触发
        assert sig.iloc[5:].all()

    def test_time_exit_not_triggered_when_profitable(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="B")
        pos = pd.Series([1] * 10, index=idx)
        pnl = pd.Series([0.05] * 10, index=idx)  # 浮盈 ≥ 阈值
        sig = time_exit_signals(pos, pnl, max_holding_days=5, min_pnl_threshold=0.0)
        assert not sig.any()

    def test_time_exit_flat_position(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="B")
        pos = pd.Series([0] * 10, index=idx)
        pnl = pd.Series([0.0] * 10, index=idx)
        sig = time_exit_signals(pos, pnl, max_holding_days=3)
        assert not sig.any()


class TestRiskManager:
    def test_circuit_breaker_triggers(self):
        cb = DrawdownCircuitBreaker(threshold=-0.10)
        navs = [100, 110, 105, 99, 95]  # 从 110 跌至 95, 回撤 -13.6%
        triggered = [cb.update(n) for n in navs]
        assert not triggered[0]
        assert triggered[-1]  # 跌至 95 时触发

    def test_circuit_breaker_recovery(self):
        # 观察期语义: 触发后锁 observation_days, 期满自动解除 (峰值重置)
        cb = DrawdownCircuitBreaker(threshold=-0.10, observation_days=3)
        cb.update(100)
        cb.update(120)
        cb.update(107)  # 回撤 ~-10.8% → 触发
        assert cb.is_active
        # 观察期内持续 active
        cb.update(115)
        cb.update(118)
        assert cb.is_active
        # 观察期满 (第3次更新) → 解除
        cb.update(119)
        assert not cb.is_active

    def test_circuit_breaker_observation_retrigger(self):
        # 观察期解除后若继续回撤 → 再次触发 (峰值已重置, 相对新峰值)
        cb = DrawdownCircuitBreaker(threshold=-0.10, observation_days=1)
        cb.update(100)
        cb.update(89)   # -11% → 触发
        assert cb.is_active
        cb.update(88)   # 观察期满 → 解除, peak=88
        assert not cb.is_active
        cb.update(79)   # 相对 88 回撤 -10.2% → 再次触发
        assert cb.is_active

    def test_target_vol_leverage_bounded(self):
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0, 0.01, 100))
        lev = target_vol_leverage(rets, target_vol=0.15, vol_lookback=20, max_leverage=3.0)
        assert (lev <= 3.0).all()
        assert (lev >= 0).all()

    def test_rolling_drawdown(self):
        nav = pd.Series([100, 110, 120, 90, 100], index=pd.date_range("2025-01-01", periods=5))
        dd = rolling_drawdown(nav)
        assert dd.iloc[2] == 0.0  # 创新高
        assert abs(dd.iloc[3] - (-0.25)) < 1e-9  # 从 120 跌至 90
        assert dd.iloc[4] < 0  # 仍未回到 120

    def test_stress_test(self):
        rng = np.random.RandomState(42)
        rets = pd.Series(np.concatenate([rng.normal(0, 0.01, 99), [0.10]]))
        result = stress_test_extreme_days(rets, quantile=0.99, metric="calmar")
        assert result["extreme_days"] >= 1
        assert result["full"] is not None or np.isnan(result["full"])


class TestExitManager:
    def _make_inputs(self):
        """构造 evaluate_exit 测试输入."""
        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        ohlc = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1, "close": close,
        }, index=idx)
        # 持续持多 30 天后平仓
        position = pd.Series([1] * 30 + [0] * 20, index=idx)
        # 状态: trend_up 持续 → 25 日后变 range
        states = pd.DataFrame({
            "A": ["trend_up"] * 25 + ["range"] * 25,
        }, index=idx)
        # 浮盈: 早期为正 (赚钱), 后转负 (被套)
        pnl = pd.Series(np.concatenate([
            np.linspace(0.02, 0.05, 25),
            np.linspace(-0.01, -0.05, 25),
        ]), index=idx)
        return ohlc, position, states, pnl

    def test_parallel_combination(self):
        layers = ExitLayers(
            technical=pd.Series([False, True, False, False]),
            logical=pd.Series([False, False, True, False]),
            time_based=pd.Series([True, False, False, False]),
            risk_halt=pd.Series([False, False, False, False]),
        )
        result = combine_exit_signals(layers, mode="parallel")
        assert result.tolist() == [True, True, True, False]

    def test_serial_combination(self):
        layers = ExitLayers(
            technical=pd.Series([False, True, False, False]),
            logical=pd.Series([False, False, True, False]),
            time_based=pd.Series([True, False, False, False]),
            risk_halt=pd.Series([False, False, False, False]),
        )
        result = combine_exit_signals(layers, mode="serial")
        assert not result.any()

    def test_evaluate_exit_integration(self):
        ohlc, position, states, pnl = self._make_inputs()
        # 状态反转 + 时间达到 + 后期亏损 → 应触发退出
        sig = evaluate_exit(
            position=position,
            ohlc=ohlc,
            states=states,
            unrealized_pnl=pnl,
            max_holding_days=20,
            chandelier_kwargs={"window": 10, "atr_window": 10, "mult": 2.0},
            logical_confirm_window=3,
            mode="parallel",
        )
        # 至少应有一些退出信号
        assert sig.any()

    def test_evaluate_exit_serial(self):
        ohlc, position, states, pnl = self._make_inputs()
        # 串联: 要求所有层同时触发, 通常更稀疏
        sig_serial = evaluate_exit(
            position=position,
            ohlc=ohlc,
            states=states,
            unrealized_pnl=pnl,
            max_holding_days=20,
            chandelier_kwargs={"window": 10, "atr_window": 10, "mult": 2.0},
            logical_confirm_window=3,
            mode="serial",
        )
        sig_parallel = evaluate_exit(
            position=position,
            ohlc=ohlc,
            states=states,
            unrealized_pnl=pnl,
            max_holding_days=20,
            chandelier_kwargs={"window": 10, "atr_window": 10, "mult": 2.0},
            logical_confirm_window=3,
            mode="parallel",
        )
        # 串联合并信号 ≤ 并联
        assert sig_serial.sum() <= sig_parallel.sum()

    def test_exit_reason_attribution(self):
        layers = ExitLayers(
            technical=pd.Series([True, False, False, False]),
            logical=pd.Series([False, True, False, False]),
            time_based=pd.Series([False, False, True, False]),
            risk_halt=pd.Series([False, False, False, False]),
        )
        reasons = exit_reason(layers)
        assert reasons.iloc[0] == "technical"
        assert reasons.iloc[1] == "logical"
        assert reasons.iloc[2] == "time"
        assert reasons.iloc[3] == ""
