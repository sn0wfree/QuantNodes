"""Unit tests for v10.3 schemes F (Conservative) and G (Trend-Adaptive).

Phase 2 of docs/82 roadmap.

覆盖:
- _apply_constraints: min defensive / max turnover / hysteresis / normalization
- scheme_f_conservative: produces valid weights, respects min(v1.0)>=0.25
- scheme_g_trend_adaptive: trend up → base scheme, trend down → conservative
- 共享 utility: 权重和 = 1, v1.0 >= 0.25 (F)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


class TestApplyConstraints:
    def test_min_defensive_raises_v1_weight(self) -> None:
        """v1.0 < 0.25 → 强制提到 0.25, 从其他策略扣减."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            _apply_constraints,
        )
        w = {"v1.0": 0.10, "v7.10": 0.30, "v9macro": 0.30, "DualMom": 0.30}
        prev_w = {"v1.0": 0.74, "v7.10": 0.09, "v9macro": 0.12, "DualMom": 0.05}
        result = _apply_constraints(w, prev_w)
        assert result["v1.0"] >= 0.25, f"v1.0={result['v1.0']} 应 >= 0.25"
        # 总和为 1
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_max_turnover_caps_weight_drift(self) -> None:
        """换手 > 0.30 → 缩放到 0.30."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            _apply_constraints,
        )
        w = {"v1.0": 0.10, "v7.10": 0.60, "v9macro": 0.20, "DualMom": 0.10}
        prev_w = {"v1.0": 0.74, "v7.10": 0.09, "v9macro": 0.12, "DualMom": 0.05}
        result = _apply_constraints(w, prev_w)
        turnover = sum(abs(result[k] - prev_w[k]) for k in result)
        # max_turnover 是 0.30
        assert turnover <= 0.31, f"换手 {turnover:.3f} 超过上限 0.30"

    def test_hysteresis_returns_prev_when_drift_small(self) -> None:
        """weight drift < 0.05 → 不调仓, 返回 prev_w."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            _apply_constraints,
        )
        w = {"v1.0": 0.75, "v7.10": 0.10, "v9macro": 0.10, "DualMom": 0.05}
        prev_w = {"v1.0": 0.74, "v7.10": 0.09, "v9macro": 0.12, "DualMom": 0.05}
        result = _apply_constraints(w, prev_w)
        # 应该返回 prev_w (drift < 0.05)
        assert result == prev_w, f"drift 小应不调仓, 但 {result} != {prev_w}"

    def test_negative_weights_clipped_to_zero(self) -> None:
        """负权重 → 0, 然后归一化."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            _apply_constraints,
        )
        w = {"v1.0": 0.50, "v7.10": -0.10, "v9macro": 0.30, "DualMom": 0.30}
        prev_w = {"v1.0": 0.74, "v7.10": 0.09, "v9macro": 0.12, "DualMom": 0.05}
        result = _apply_constraints(w, prev_w)
        for v in result.values():
            assert v >= 0, f"权重不能为负: {result}"
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestSchemeFConservative:
    def test_min_v1_weight_respected(self) -> None:
        """F scheme: v1.0 权重在所有日期 >= 0.25."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_f_conservative,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_f_conservative(prices)
        # v1.0 权重在 252 天后 (前 252 天是 warmup) 必须 >= 0.25
        v1_post_warmup = weights["v1.0"].iloc[252:]
        assert (v1_post_warmup >= 0.25 - 0.01).all(), \
            f"v1.0 < 0.25 出现: min={v1_post_warmup.min()}"

    def test_weights_sum_to_one(self) -> None:
        """权重总和 = 1."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_f_conservative,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_f_conservative(prices)
        sums = weights.sum(axis=1)
        # 容差 0.01
        assert ((sums - 1.0).abs() < 0.01).all(), \
            f"权重和不=1: max diff={(sums - 1.0).abs().max()}"

    def test_no_negative_weights(self) -> None:
        """F scheme 没有负权重."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_f_conservative,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_f_conservative(prices)
        assert (weights >= 0).all().all(), "F scheme 有负权重"

    def test_monthly_rebalance(self) -> None:
        """F scheme 月频调仓 (rebal_dates = 月末)."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_f_conservative,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_f_conservative(prices)
        # 每日权重的变化数应 < 252 (vs 方案 D 是 252-300+ 次)
        # F 月频 → ~84 月 (2018-2026) 调仓次数
        # 计算连续 2 天的 weight 变化 > 0.01 的次数
        diffs = weights.diff().abs().sum(axis=1)
        n_changes = (diffs > 0.01).sum()
        # 月频调仓 + warmup = 84 + 1
        assert n_changes <= 100, f"F scheme 调仓次数 {n_changes} 超过月频预期"


class TestSchemeGTrendAdaptive:
    def test_weights_sum_to_one(self) -> None:
        """G scheme 权重总和 = 1."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_g_trend_adaptive,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_g_trend_adaptive(prices)
        sums = weights.sum(axis=1)
        assert ((sums - 1.0).abs() < 0.01).all(), \
            f"权重和不=1: max diff={(sums - 1.0).abs().max()}"

    def test_no_negative_weights(self) -> None:
        """G scheme 没有负权重."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_g_trend_adaptive,
            load_navs,
        )
        prices = load_navs()
        weights = scheme_g_trend_adaptive(prices)
        assert (weights >= 0).all().all(), "G scheme 有负权重"

    def test_trend_filter_changes_weights(self) -> None:
        """G scheme 趋势 up 时权重与 F 不同 (用 base scheme)."""
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            scheme_f_conservative,
            scheme_g_trend_adaptive,
            load_navs,
        )
        prices = load_navs()
        w_f = scheme_f_conservative(prices)
        w_g = scheme_g_trend_adaptive(prices)
        # 至少在 5%+ 的日期, F 和 G 权重不同 (G 顺势时不同)
        diffs = (w_f - w_g).abs().sum(axis=1)
        n_diff = (diffs > 0.01).sum()
        assert n_diff > 0.05 * len(w_f), \
            f"G 与 F 不同的天数 {n_diff} < 5% 预期"
