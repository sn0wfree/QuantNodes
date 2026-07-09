# coding=utf-8
"""Tests for Stage 16A: 子策略权重分配 (sub_weighting_v3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v3 import (
    SubStrategyResult,
    combine_sub_results,
    equal_sub_weights,
    risk_parity_sub_weights,
    signal_weighted_sub_weights,
)


def _make_sub_result(name: str, weights: dict, signal: float = 0.5, date=None) -> SubStrategyResult:
    """构造一个 SubStrategyResult."""
    if date is None:
        date = pd.Timestamp("2024-01-15")
    return SubStrategyResult(
        date=date,
        chosen=list(weights.keys()),
        weights=weights,
        signal_strength=signal,
        meta={"strategy": name},
    )


def _make_sub_navs(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """构造 3 个子策略的 NAV 序列."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rets = rng.normal(0.0005, 0.005, size=(n_days, 3))
    # 子策略 0 波动大, 子策略 2 波动小
    rets[:, 0] *= 2.0
    rets[:, 2] *= 0.5
    navs = np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(navs, index=dates, columns=["momentum", "reversion", "industry"])


class TestEqualSubWeights:
    """equal_sub_weights 单元测试."""

    def test_basic_3_strategies(self):
        weights = equal_sub_weights(["mom", "rev", "ind"])
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0/3) < 1e-9

    def test_with_int(self):
        weights = equal_sub_weights(3)
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0/3) < 1e-9

    def test_zero_strategies(self):
        assert equal_sub_weights(0) == {}
        assert equal_sub_weights([]) == {}


class TestRiskParitySubWeights:
    """risk_parity_sub_weights 单元测试."""

    def test_basic_three_sub_strategies(self):
        navs = _make_sub_navs(n_days=120)
        weights = risk_parity_sub_weights(navs, method="sample")
        assert len(weights) == 3
        # 权重和为 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # 所有权重非负
        for w in weights.values():
            assert w >= 0

    def test_high_vol_strategy_gets_lower_weight(self):
        """高波动子策略应获得较低权重 (风险平价核心)."""
        navs = _make_sub_navs(n_days=200)
        weights = risk_parity_sub_weights(navs, method="sample")
        # momentum 波动大, 应得最低权重
        # industry 波动小, 应得最高权重
        assert weights["industry"] >= weights["momentum"]

    def test_insufficient_data(self):
        """数据不足时回退到等权."""
        navs = _make_sub_navs(n_days=3)  # 不足
        weights = risk_parity_sub_weights(navs, method="sample")
        # 应该是等权
        for w in weights.values():
            assert abs(w - 1.0/3) < 0.1  # 近似

    def test_different_cov_methods(self):
        """不同协方差方法都能用."""
        navs = _make_sub_navs(n_days=120)
        for method in ["sample", "ledoit_wolf", "ewma"]:
            weights = risk_parity_sub_weights(navs, method=method)
            assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestSignalWeightedSubWeights:
    """signal_weighted_sub_weights 单元测试."""

    def test_strong_signal_higher_weight(self):
        """强信号获得更高权重."""
        strong = _make_sub_result("momentum", {"a": 0.5, "b": 0.5}, signal=0.8)
        weak = _make_sub_result("reversion", {"c": 1.0}, signal=-0.5)
        weights = signal_weighted_sub_weights([strong, weak])
        # 验证: 强信号应得更高权重
        assert weights["momentum"] > weights["reversion"]

    def test_clip_negative_signal(self):
        """负信号被 clip 到 0."""
        very_negative = _make_sub_result("industry", {"x": 1.0}, signal=-2.0)
        weights = signal_weighted_sub_weights([very_negative], signal_clip=1.0)
        # 完全没信号 → 移到非负区间后为 0
        # 但 sum = 0, 应该 fallback 等权
        assert sum(weights.values()) > 0

    def test_empty_results(self):
        weights = signal_weighted_sub_weights([])
        assert weights == {}


class TestCombineSubResults:
    """combine_sub_results 单元测试."""

    def test_basic_combine(self):
        mom = _make_sub_result("momentum", {"a": 0.5, "b": 0.5}, signal=0.5)
        rev = _make_sub_result("reversion", {"c": 1.0}, signal=0.3)
        weights = equal_sub_weights(["momentum", "reversion"])
        combined = combine_sub_results([mom, rev], weights)
        # 应该有 3 个不同的 ETF
        assert len(combined) == 3
        # 权重和为 1
        assert abs(sum(combined.values()) - 1.0) < 1e-6

    def test_overlap_aggregation(self):
        """同一 ETF 被多个子策略选中 → 权重累加."""
        mom = _make_sub_result("momentum", {"a": 0.5, "b": 0.5}, signal=0.5)
        rev = _make_sub_result("reversion", {"a": 1.0}, signal=0.3)
        weights = equal_sub_weights(["momentum", "reversion"])
        combined = combine_sub_results([mom, rev], weights)
        # 'a' 出现在两个子策略
        # 合并: 0.5 * 0.5 + 0.5 * 1.0 = 0.25 + 0.5 = 0.75 (归一化前)
        # 归一化: 0.75 / (0.75 + 0.25) = 0.75
        assert "a" in combined
        assert combined["a"] > 0.5  # 高于任一子策略的单独权重

    def test_pool_filter(self):
        """pool_codes 过滤掉非法 ETF."""
        mom = _make_sub_result("momentum", {"a": 0.5, "b": 0.5})
        weights = equal_sub_weights(["momentum"])
        combined = combine_sub_results([mom], weights, pool_codes={"a"})
        # 'b' 应被过滤
        assert "a" in combined
        assert "b" not in combined

    def test_empty_sub_weights(self):
        """空子策略列表."""
        combined = combine_sub_results([], equal_sub_weights([]))
        assert combined == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
