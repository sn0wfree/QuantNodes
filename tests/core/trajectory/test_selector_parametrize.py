"""ParentSelector 全策略全参数 parametrize (~30 tests)。

遍历 5 策略 + 各 metric/seed/n/threshold 组合 + 异常。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import (
    ParentSelector,
    SelectionStrategy,
    TrajectoryEntry,
    TrajectoryPool,
)


def _entry(eid: str, sharpe: float, decision: bool = True) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid,
        feedback=FactorFeedback(
            factor_id=eid, factor_name=f"f_{eid}", decision=decision,
        ),
        metrics={"sharpe": sharpe},
    )


@pytest.fixture
def pool_with_10(tmp_path: Path) -> TrajectoryPool:
    pool = TrajectoryPool(tmp_path)
    for i in range(10):
        pool.add(_entry(f"e{i:02d}", float(i)))
    return pool


# ============================================================================
# 1. Construction 参数 (6 tests)
# ============================================================================

class TestConstruction:
    @pytest.mark.parametrize("strategy,metric,threshold,seed", [
        ("best", "sharpe", 0.3, 42),
        ("random", "ic_mean", 0.5, None),
        ("weighted", "sharpe", 0.3, 0),
        ("weighted_inverse", "arr", 0.7, 100),
        ("top_percent_plus_random", "calmar", 0.1, 1),
    ])
    def test_valid_construction(self, strategy, metric, threshold, seed):
        s = ParentSelector(strategy=strategy, metric=metric,
                           top_percent_threshold=threshold, seed=seed)
        assert s.strategy == strategy
        assert s.metric == metric

    @pytest.mark.parametrize("bad_strategy", [
        "unknown", "", "BEST", "Best", None, "weighted_inverse_typo",
    ])
    def test_invalid_strategy_raises(self, bad_strategy):
        with pytest.raises(ValueError, match="未知 strategy"):
            ParentSelector(strategy=bad_strategy)

    @pytest.mark.parametrize("metric_name", [
        "sharpe", "ic_mean", "rank_ic_mean", "arr", "mdd", "calmar", "ic_ir",
    ])
    def test_all_known_metrics(self, metric_name):
        s = ParentSelector(metric=metric_name)
        assert s.metric == metric_name


# ============================================================================
# 2. select n 参数 (8 tests)
# ============================================================================

class TestSelectN:
    @pytest.mark.parametrize("n,expected_len", [
        (0, 0),
        (1, 1),
        (3, 3),
        (10, 10),
        (15, 10),  # n > pool
    ])
    def test_n_variants(self, pool_with_10, n, expected_len):
        s = ParentSelector(strategy="best")
        r = s.select(pool_with_10, n=n)
        assert len(r) == expected_len


# ============================================================================
# 3. 5 策略 × 多 metric (15 tests)
# ============================================================================

class TestStrategyMetrics:
    @pytest.mark.parametrize("strategy", [
        "best", "random", "weighted", "weighted_inverse", "top_percent_plus_random",
    ])
    def test_all_strategies_run(self, pool_with_10, strategy):
        s = ParentSelector(strategy=strategy, seed=42)
        r = s.select(pool_with_10, n=3)
        assert len(r) == 3
        # 都是 pool 中的 entry
        ids = {e.entry_id for e in r}
        assert all(i.startswith("e") for i in ids)

    @pytest.mark.parametrize("strategy,expected_order", [
        ("best", ["e09", "e08", "e07"]),  # 高 → 低
        ("random", None),  # 不确定
    ])
    def test_strategy_order(self, pool_with_10, strategy, expected_order):
        s = ParentSelector(strategy=strategy, seed=42)
        r = s.select(pool_with_10, n=3)
        if expected_order is not None:
            actual = [e.entry_id for e in r]
            assert actual == expected_order

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.9, 1.0])
    def test_top_percent_thresholds(self, pool_with_10, threshold):
        s = ParentSelector(strategy="top_percent_plus_random", top_percent_threshold=threshold, seed=42)
        r = s.select(pool_with_10, n=3)
        assert len(r) == 3


# ============================================================================
# 4. seed 决定性 (4 tests)
# ============================================================================

class TestSeedDeterminism:
    def test_random_same_seed_same_result(self, pool_with_10):
        s1 = ParentSelector(strategy="random", seed=42)
        s2 = ParentSelector(strategy="random", seed=42)
        r1 = {e.entry_id for e in s1.select(pool_with_10, n=3)}
        r2 = {e.entry_id for e in s2.select(pool_with_10, n=3)}
        assert r1 == r2

    def test_weighted_same_seed(self, pool_with_10):
        s1 = ParentSelector(strategy="weighted", seed=42)
        s2 = ParentSelector(strategy="weighted", seed=42)
        r1 = {e.entry_id for e in s1.select(pool_with_10, n=3)}
        r2 = {e.entry_id for e in s2.select(pool_with_10, n=3)}
        assert r1 == r2

    def test_different_seed_different_result(self, pool_with_10):
        """高概率不同。"""
        s1 = ParentSelector(strategy="random", seed=1)
        s2 = ParentSelector(strategy="random", seed=2)
        r1 = [e.entry_id for e in s1.select(pool_with_10, n=10)]
        r2 = [e.entry_id for e in s2.select(pool_with_10, n=10)]
        # 至少 1 个位置不同
        diffs = sum(1 for a, b in zip(r1, r2) if a != b)
        assert diffs > 0

    def test_none_seed_random(self, pool_with_10):
        """seed=None 不崩。"""
        s = ParentSelector(strategy="random", seed=None)
        r = s.select(pool_with_10, n=3)
        assert len(r) == 3


# ============================================================================
# 5. pool 边界 (5 tests)
# ============================================================================

class TestPoolEdges:
    @pytest.mark.parametrize("strategy", [
        "best", "random", "weighted", "weighted_inverse", "top_percent_plus_random",
    ])
    def test_empty_pool(self, tmp_path: Path, strategy):
        pool = TrajectoryPool(tmp_path)
        s = ParentSelector(strategy=strategy)
        assert s.select(pool, n=5) == []

    @pytest.mark.parametrize("strategy", [
        "best", "random", "weighted", "weighted_inverse", "top_percent_plus_random",
    ])
    def test_all_rejected(self, tmp_path: Path, strategy):
        """所有 entry decision=False → 返回 []."""
        pool = TrajectoryPool(tmp_path)
        for i in range(5):
            pool.add(_entry(f"e{i}", 0.5, decision=False))
        s = ParentSelector(strategy=strategy)
        assert s.select(pool, n=3) == []

    def test_single_entry(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("only", 1.0))
        s = ParentSelector(strategy="best")
        r = s.select(pool, n=3)
        assert len(r) == 1

    def test_list_pool_input(self, tmp_path: Path):
        """接受 list。"""
        pool = TrajectoryPool(tmp_path)
        for i in range(5):
            pool.add(_entry(f"e{i}", float(i)))
        s = ParentSelector(strategy="best")
        r = s.select(list(pool.all()), n=2)
        assert len(r) == 2

    def test_top_percent_1_n_larger(self, tmp_path: Path):
        """threshold=1.0, n 大于 pool, 全部 top。"""
        pool = TrajectoryPool(tmp_path)
        for i in range(3):
            pool.add(_entry(f"e{i}", float(i)))
        s = ParentSelector(strategy="top_percent_plus_random", top_percent_threshold=1.0, seed=42)
        r = s.select(pool, n=10)
        assert len(r) == 3


# ============================================================================
# 6. metric 边界 (3 tests)
# ============================================================================

class TestMetricEdges:
    @pytest.mark.parametrize("metric_value", [0.0, 0.5, 1.0, -1.0, 100.0])
    def test_metric_with_various_values(self, tmp_path: Path, metric_value):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", metric_value))
        pool.add(_entry("e2", metric_value + 0.1))
        s = ParentSelector(strategy="best", metric="sharpe")
        r = s.select(pool, n=1)
        assert r[0].entry_id in {"e1", "e2"}

    def test_missing_metric_key(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(factor_id="e1", factor_name="f", decision=True),
            metrics={},  # 无 sharpe
        ))
        s = ParentSelector(strategy="best", metric="sharpe")
        r = s.select(pool, n=1)
        assert r[0].entry_id == "e1"  # 当 0 处理, 仍返回

    def test_nan_metric_does_not_crash(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        import math
        pool.add(TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(factor_id="e1", factor_name="f", decision=True),
            metrics={"sharpe": float("nan")},
        ))
        s = ParentSelector(strategy="best", metric="sharpe")
        r = s.select(pool, n=1)
        # 不崩
        assert len(r) == 1
