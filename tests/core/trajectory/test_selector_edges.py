"""selector.py 边界条件测试 (20 tests)。

聚焦:
    - 5 种策略的边界: 空池、单元素、n 超过池大小
    - 权重计算: NaN/Inf/全 0 -> 降级到 random
    - 反向选择 (weighted_inverse) 与正向的区别
    - 未知 strategy 抛 ValueError
    - 接受 list / TrajectoryPool 两种 pool 形式
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


def _entry(entry_id: str, sharpe: float, decision: bool = True) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id,
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=f"f_{entry_id}",
            decision=decision, summary=f"sharpe={sharpe}",
        ) if decision else None,
        metrics={"sharpe": sharpe} if decision else {},
    )


def _pool_with(entries: list[TrajectoryEntry], tmp_path: Path) -> TrajectoryPool:
    pool = TrajectoryPool(tmp_path)
    for e in entries:
        pool.add(e)
    return pool


# ============================================================================
# 1. 5 种策略 + 构造验证 (8 tests)
# ============================================================================

class TestSelectorConstruction:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="未知 strategy"):
            ParentSelector(strategy="bogus")

    def test_default_strategy_is_best(self):
        s = ParentSelector()
        assert s.strategy == "best"
        assert s.metric == "sharpe"

    def test_strategy_enum_values(self):
        assert SelectionStrategy.BEST.value == "best"
        assert SelectionStrategy.RANDOM.value == "random"
        assert SelectionStrategy.WEIGHTED.value == "weighted"
        assert SelectionStrategy.WEIGHTED_INVERSE.value == "weighted_inverse"
        assert SelectionStrategy.TOP_PERCENT_PLUS_RANDOM.value == "top_percent_plus_random"


class TestBestStrategy:
    def test_best_returns_top_n(self, tmp_path: Path):
        entries = [_entry(f"e{i}", float(i)) for i in range(5)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="best")
        result = s.select(pool, n=2)
        assert len(result) == 2
        # best = e4 (sharpe=4.0)
        assert result[0].entry_id == "e4"

    def test_best_n_larger_than_pool(self, tmp_path: Path):
        entries = [_entry(f"e{i}", float(i)) for i in range(3)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="best")
        result = s.select(pool, n=10)
        assert len(result) == 3  # 池只有 3 个


class TestRandomStrategy:
    def test_random_reproducible(self, tmp_path: Path):
        """两个相同 seed 的 selector 返回相同集合。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(10)]
        pool = _pool_with(entries, tmp_path)
        s1 = ParentSelector(strategy="random", seed=42)
        s2 = ParentSelector(strategy="random", seed=42)
        r1 = s1.select(pool, n=3)
        r2 = s2.select(pool, n=3)
        # 不同实例相同 seed → 相同内容
        assert {e.entry_id for e in r1} == {e.entry_id for e in r2}

    def test_random_does_not_repeat(self, tmp_path: Path):
        """同一次 select 不重复 (replace=False)。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(5)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="random", seed=42)
        r = s.select(pool, n=3)
        ids = [e.entry_id for e in r]
        assert len(set(ids)) == 3  # 无重复


class TestWeightedStrategy:
    def test_weighted_prefers_high_score(self, tmp_path: Path):
        """高 sharpe 选更多。"""
        # 构造 1000 次抽样, 检查高 sharpe 占比
        high_count = 0
        for seed in range(50):
            entries = [_entry(f"low_{i}", 0.1) for i in range(5)] + \
                      [_entry("high_0", 10.0)]
            pool = _pool_with(entries, tmp_path)
            s = ParentSelector(strategy="weighted", seed=seed)
            r = s.select(pool, n=3)
            if any(e.entry_id == "high_0" for e in r):
                high_count += 1
        # 高 sharpe 应被抽到超过半数
        assert high_count > 25

    def test_weighted_degrades_to_random_on_zero_weight(self, tmp_path: Path):
        """全 0 metrics -> exp(0) = 1 -> 仍能选, 不崩。"""
        entries = [_entry(f"e{i}", 0.0) for i in range(5)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="weighted", seed=42)
        r = s.select(pool, n=2)
        assert len(r) == 2  # 不崩


class TestWeightedInverse:
    def test_inverse_prefers_low_score(self, tmp_path: Path):
        """weighted_inverse 偏好低 sharpe (探索)。"""
        high_count = 0
        for seed in range(50):
            entries = [_entry(f"low_{i}", 0.1) for i in range(5)] + \
                      [_entry("high_0", 10.0)]
            pool = _pool_with(entries, tmp_path)
            s = ParentSelector(strategy="weighted_inverse", seed=seed)
            r = s.select(pool, n=3)
            if any(e.entry_id == "high_0" for e in r):
                high_count += 1
        # inverse 反而 low 被选更多, high 较少
        assert high_count < 25


class TestTopPercentPlusRandom:
    def test_top_percent_includes_top_n(self, tmp_path: Path):
        """top 30% 应包含高分 entry。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(10)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(
            strategy="top_percent_plus_random", top_percent_threshold=0.3, seed=42,
        )
        r = s.select(pool, n=3)
        # 30% of 10 = 3 → 全部是 top 3
        assert {e.entry_id for e in r} == {"e7", "e8", "e9"}

    def test_top_percent_with_extra(self, tmp_path: Path):
        """n > top 30% 包含 top + 随机。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(10)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(
            strategy="top_percent_plus_random", top_percent_threshold=0.2, seed=42,
        )
        r = s.select(pool, n=5)  # top 2 + random 3
        assert len(r) == 5
        # 前 2 必须是 top
        top_2_ids = {e.entry_id for e in r if e.entry_id in {"e9", "e8"}}
        assert len(top_2_ids) == 2


# ============================================================================
# 2. 边界 + 异常 (8 tests)
# ============================================================================

class TestSelectorEdges:
    def test_empty_pool_returns_empty(self, tmp_path: Path):
        pool = _pool_with([], tmp_path)
        for strat in ["best", "random", "weighted", "weighted_inverse", "top_percent_plus_random"]:
            s = ParentSelector(strategy=strat, seed=42)
            assert s.select(pool, n=3) == [], f"{strat} should return []"

    def test_pool_all_rejected_returns_empty(self, tmp_path: Path):
        """所有 entry decision=False → 过滤为空 → 返回 []。"""
        entries = [_entry(f"e{i}", 0.0, decision=False) for i in range(5)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="best")
        assert s.select(pool, n=2) == []

    def test_single_entry_returns_it(self, tmp_path: Path):
        entries = [_entry("only", 1.0)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="best")
        r = s.select(pool, n=3)
        assert len(r) == 1
        assert r[0].entry_id == "only"

    def test_n_zero_returns_empty(self, tmp_path: Path):
        entries = [_entry(f"e{i}", float(i)) for i in range(5)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(strategy="best")
        assert s.select(pool, n=0) == []

    def test_accepts_list_pool(self, tmp_path: Path):
        """接受 list[TrajectoryEntry] 而非 TrajectoryPool。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(5)]
        s = ParentSelector(strategy="best")
        r = s.select(entries, n=2)
        assert len(r) == 2
        assert r[0].entry_id == "e4"

    def test_metric_with_missing_key(self, tmp_path: Path):
        """metrics 缺 key → 当作 0 处理。"""
        e1 = TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(factor_id="e1", factor_name="f1", decision=True),
            metrics={},  # 无 sharpe
        )
        e2 = TrajectoryEntry(
            entry_id="e2",
            feedback=FactorFeedback(factor_id="e2", factor_name="f2", decision=True),
            metrics={"sharpe": 1.5},
        )
        pool = _pool_with([e1, e2], tmp_path)
        s = ParentSelector(strategy="best", metric="sharpe")
        r = s.select(pool, n=1)
        assert r[0].entry_id == "e2"  # e2 sharpe=1.5 > e1 缺省 0

    def test_metric_with_nan_does_not_crash(self, tmp_path: Path):
        """NaN metric 不会让 selector 崩溃。"""
        e1 = TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(factor_id="e1", factor_name="f1", decision=True),
            metrics={"sharpe": float("nan")},
        )
        e2 = TrajectoryEntry(
            entry_id="e2",
            feedback=FactorFeedback(factor_id="e2", factor_name="f2", decision=True),
            metrics={"sharpe": 1.0},
        )
        pool = _pool_with([e1, e2], tmp_path)
        s = ParentSelector(strategy="best", metric="sharpe")
        r = s.select(pool, n=1)
        # 不崩 + 仍能选一个
        assert len(r) == 1
        assert r[0].entry_id in {"e1", "e2"}

    def test_top_percent_threshold_edge(self, tmp_path: Path):
        """top_percent_threshold=1.0 → 全部算 top, n>pool_size 取 min(n, pool)。"""
        entries = [_entry(f"e{i}", float(i)) for i in range(3)]
        pool = _pool_with(entries, tmp_path)
        s = ParentSelector(
            strategy="top_percent_plus_random", top_percent_threshold=1.0, seed=42,
        )
        r = s.select(pool, n=5)  # n=5 > pool=3
        # 全部 top, 不足 random 数量, 返回 top
        assert len(r) == 3
