"""TrajectoryPool 模块测试 (30 tests)。

覆盖:
    - TrajectoryEntry 基础 (3)
    - TrajectoryPool CRUD (7)
    - 过滤 / 选择 API (5)
    - 5 种选择策略 (5)
    - 谱系 (5)
    - 持久化 + 重载 (3)
    - 边界 (2)
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
)
from QuantNodes.core.trajectory import (
    ParentSelector,
    SelectionStrategy,
    TrajectoryEntry,
    TrajectoryPool,
    children_of,
    descendants,
    lineage,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_feedback(name: str = "f", decision: bool = True) -> FactorFeedback:
    return FactorFeedback(factor_name=name, decision=decision, summary="ok")


def _make_entry(
    eid: str,
    round_idx: int = 0,
    operation: str = "original",
    parent_ids: list[str] | None = None,
    metrics: dict | None = None,
    decision: bool = True,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid,
        round_idx=round_idx,
        operation=operation,
        parent_ids=parent_ids or [],
        feedback=_make_feedback(name=eid, decision=decision),
        metrics=metrics or {},
    )


@pytest.fixture
def pool(tmp_path):
    return TrajectoryPool(tmp_path)


# ============================================================================
# 1. TrajectoryEntry 基础 (3)
# ============================================================================

def test_entry_dataclass_basic():
    """TrajectoryEntry 基础创建。"""
    e = TrajectoryEntry()
    assert len(e.entry_id) > 0
    assert e.round_idx == 0
    assert e.operation == "original"
    assert e.parent_ids == []
    assert e.feedback is None


def test_entry_to_parquet_row():
    """Parquet row 包含所有指标列。"""
    fb = FactorFeedback(
        factor_name="momentum",
        decision=True,
        summary="通过",
        duration_ms=123.4,
    )
    e = TrajectoryEntry(
        round_idx=1,
        operation="mutation",
        feedback=fb,
        metrics={"sharpe": 1.2, "ic_mean": 0.05, "mdd": -0.08},
    )
    row = e.to_parquet_row()
    assert row["operation"] == "mutation"
    assert row["decision"] is True
    assert row["factor_name"] == "momentum"
    assert row["sharpe"] == 1.2
    assert row["ic_mean"] == 0.05
    assert row["mdd"] == -0.08
    assert row["duration_ms"] == 123.4


def test_entry_to_from_json_dict():
    """JSON 双向序列化。"""
    fb = FactorFeedback(
        factor_name="j",
        channels={FeedbackChannel.CODE: ChannelFeedback(
            FeedbackChannel.CODE, True, "ok", 1.0,
        )},
        decision=True,
    )
    e = TrajectoryEntry(
        round_idx=2,
        operation="crossover",
        config_snapshot={"factor": {"name": "j"}},
        context_subset={"foo": "bar"},
        feedback=fb,
        parent_ids=["p1", "p2"],
        metrics={"sharpe": 2.0},
    )
    d = e.to_json_dict()
    json.dumps(d)
    e2 = TrajectoryEntry.from_json_dict(d)
    assert e2.entry_id == e.entry_id
    assert e2.parent_ids == ["p1", "p2"]
    assert e2.feedback.channels[FeedbackChannel.CODE].detail == "ok"
    assert e2.metrics["sharpe"] == 2.0


# ============================================================================
# 2. TrajectoryPool CRUD (7)
# ============================================================================

def test_pool_create_empty(pool):
    """空 pool 初始化。"""
    assert pool.size == 0
    assert pool.all() == []
    assert len(pool) == 0


def test_pool_add_single(pool):
    """添加单条。"""
    e = _make_entry("a")
    pool.add(e)
    assert pool.size == 1
    assert pool.get("a") is e


def test_pool_add_multiple(pool):
    """批量添加。"""
    for i in range(5):
        pool.add(_make_entry(f"e{i}"))
    assert pool.size == 5
    assert {e.entry_id for e in pool.all()} == {f"e{i}" for i in range(5)}


def test_pool_by_round(pool):
    """按 round 过滤。"""
    pool.add(_make_entry("a", round_idx=0))
    pool.add(_make_entry("b", round_idx=1))
    pool.add(_make_entry("c", round_idx=1))
    assert {e.entry_id for e in pool.by_round(1)} == {"b", "c"}


def test_pool_by_operation(pool):
    """按 operation 过滤。"""
    pool.add(_make_entry("a", operation="original"))
    pool.add(_make_entry("b", operation="mutation"))
    pool.add(_make_entry("c", operation="crossover"))
    assert {e.entry_id for e in pool.by_operation("mutation")} == {"b"}


def test_pool_reset(pool):
    """重置清空。"""
    pool.add(_make_entry("a"))
    pool.reset()
    assert pool.size == 0
    assert not (pool.base_dir / "trajectories.parquet").exists()


def test_pool_size_property(pool):
    """size 属性同步。"""
    assert pool.size == 0
    pool.add(_make_entry("a"))
    assert pool.size == 1
    pool.add(_make_entry("b"))
    assert pool.size == 2


# ============================================================================
# 3. 过滤 / 选择 API (5)
# ============================================================================

def test_pool_best_ordering(pool):
    """best 按 sharpe 降序。"""
    pool.add(_make_entry("low", metrics={"sharpe": 0.1}))
    pool.add(_make_entry("high", metrics={"sharpe": 2.0}))
    pool.add(_make_entry("mid", metrics={"sharpe": 1.0}))
    top = pool.best(top_n=2, metric="sharpe")
    assert [e.entry_id for e in top] == ["high", "mid"]


def test_pool_best_empty_metric(pool):
    """缺指标时用 0。"""
    pool.add(_make_entry("a", metrics={}))
    pool.add(_make_entry("b", metrics={"sharpe": 1.0}))
    top = pool.best(top_n=1, metric="sharpe")
    assert top[0].entry_id == "b"


def test_pool_filter_decision_true(pool):
    """filter(decision=True) 过滤失败。"""
    pool.add(_make_entry("pass", decision=True))
    pool.add(_make_entry("fail", decision=False))
    pool.add(_make_entry("pass2", decision=True))
    ids = {e.entry_id for e in pool.filter(decision=True)}
    assert ids == {"pass", "pass2"}


def test_pool_filter_decision_false(pool):
    """filter(decision=False) 过滤失败。"""
    pool.add(_make_entry("pass", decision=True))
    pool.add(_make_entry("fail", decision=False))
    ids = {e.entry_id for e in pool.filter(decision=False)}
    assert ids == {"fail"}


def test_pool_random_n(pool):
    """random 抽 n 条不重复。"""
    for i in range(10):
        pool.add(_make_entry(f"e{i}"))
    picked = pool.random(n=5, seed=42)
    assert len(picked) == 5
    assert len({e.entry_id for e in picked}) == 5


# ============================================================================
# 4. 5 种选择策略 (5)
# ============================================================================

def _build_pool_for_selector() -> list[TrajectoryEntry]:
    return [
        _make_entry("e1", metrics={"sharpe": 0.1, "arr": 0.05}),
        _make_entry("e2", metrics={"sharpe": 2.0, "arr": 0.20}),
        _make_entry("e3", metrics={"sharpe": 1.0, "arr": 0.10}),
        _make_entry("e4", metrics={"sharpe": 1.5, "arr": 0.15}),
        _make_entry("e5", metrics={"sharpe": 0.5, "arr": 0.08}),
    ]


def test_selector_best():
    """best 策略: 选 sharpe 最高的 n 个。"""
    sel = ParentSelector(strategy="best", metric="sharpe")
    result = sel.select(_build_pool_for_selector(), n=2)
    assert [e.entry_id for e in result] == ["e2", "e4"]


def test_selector_random_distribution():
    """random 抽 5 次, 至少 2 个不同 (统计性检查)。"""
    sel = ParentSelector(strategy="random", seed=42)
    pool = _build_pool_for_selector()
    results = {tuple(sorted(e.entry_id for e in sel.select(pool, n=2))) for _ in range(20)}
    assert len(results) >= 2


def test_selector_weighted_distribution():
    """weighted: 高 sharpe 选中概率高。"""
    sel = ParentSelector(strategy="weighted", metric="sharpe", seed=42)
    pool = _build_pool_for_selector()
    counts: dict[str, int] = {e.entry_id: 0 for e in pool}
    for _ in range(200):
        result = sel.select(pool, n=1)
        counts[result[0].entry_id] += 1
    # e2 (sharpe=2.0) 应被选中最多次
    assert counts["e2"] > counts["e1"]


def test_selector_weighted_inverse():
    """weighted_inverse: 低 sharpe 选中概率高。"""
    sel = ParentSelector(strategy="weighted_inverse", metric="sharpe", seed=42)
    pool = _build_pool_for_selector()
    counts: dict[str, int] = {e.entry_id: 0 for e in pool}
    for _ in range(200):
        result = sel.select(pool, n=1)
        counts[result[0].entry_id] += 1
    # e1 (sharpe=0.1) 应被选中最多次
    assert counts["e1"] > counts["e2"]


def test_selector_top_percent_plus_random():
    """top_percent_plus_random: top 30% + 随机补足。"""
    sel = ParentSelector(
        strategy="top_percent_plus_random",
        metric="sharpe",
        top_percent_threshold=0.4,
        seed=42,
    )
    pool = _build_pool_for_selector()
    result = sel.select(pool, n=4)
    assert len(result) == 4
    # top 30% (≈2 个) 应是 sharpe 最高的
    top_ids = {e.entry_id for e in result[:2]}
    assert "e2" in top_ids and "e4" in top_ids


# ============================================================================
# 5. 谱系 (5)
# ============================================================================

def test_children_of_single(pool):
    """单子代。"""
    pool.add(_make_entry("p"))
    pool.add(_make_entry("c", parent_ids=["p"]))
    children = pool.children_of("p")
    assert [e.entry_id for e in children] == ["c"]


def test_children_of_multiple(pool):
    """多子代 (crossover: 多 parent)。"""
    pool.add(_make_entry("a"))
    pool.add(_make_entry("b"))
    pool.add(_make_entry("c", parent_ids=["a", "b"]))
    assert {e.entry_id for e in pool.children_of("a")} == {"c"}
    assert {e.entry_id for e in pool.children_of("b")} == {"c"}


def test_lineage_chain(pool):
    """单链谱系 (A0 → A1 → A2)。"""
    pool.add(_make_entry("A0", round_idx=0))
    pool.add(_make_entry("A1", round_idx=1, parent_ids=["A0"]))
    pool.add(_make_entry("A2", round_idx=2, parent_ids=["A1"]))
    chain = pool.lineage("A2")
    assert [e.entry_id for e in chain] == ["A0", "A1", "A2"]


def test_lineage_branch():
    """树状谱系 (crossover: B1 + A1 → C2)。"""
    entries = {
        "A0": _make_entry("A0", round_idx=0),
        "A1": _make_entry("A1", round_idx=1, parent_ids=["A0"]),
        "B0": _make_entry("B0", round_idx=0),
        "B1": _make_entry("B1", round_idx=1, parent_ids=["B0"]),
        "C2": _make_entry("C2", round_idx=2, parent_ids=["B1", "A1"]),
    }
    chain = lineage(entries, "C2")
    assert [e.entry_id for e in chain] == ["B0", "B1", "C2"]


def test_lineage_orphan(pool):
    """孤儿节点 (无 parent) 返回自己。"""
    pool.add(_make_entry("o"))
    chain = pool.lineage("o")
    assert [e.entry_id for e in chain] == ["o"]


def test_descendants_depth():
    """descendants 返回所有后代, BFS。"""
    entries = {
        "root": _make_entry("root"),
        "c1": _make_entry("c1", parent_ids=["root"]),
        "c2": _make_entry("c2", parent_ids=["root"]),
        "gc": _make_entry("gc", parent_ids=["c1"]),
    }
    desc = descendants(entries, "root")
    ids = {e.entry_id for e in desc}
    assert ids == {"c1", "c2", "gc"}


# ============================================================================
# 6. 持久化 + 重载 (3)
# ============================================================================

def test_persist_reload_roundtrip(tmp_path):
    """重载后 size / entry 一致。"""
    pool = TrajectoryPool(tmp_path)
    pool.add(_make_entry("a", round_idx=0, metrics={"sharpe": 1.0}))
    pool.add(_make_entry("b", round_idx=1, operation="mutation",
                        parent_ids=["a"], metrics={"sharpe": 2.0}))

    pool2 = TrajectoryPool(tmp_path)
    assert pool2.size == 2
    assert pool2.get("a").metrics["sharpe"] == 1.0
    assert pool2.get("b").parent_ids == ["a"]
    assert pool2.lineage("b")[-1].entry_id == "b"


def test_parquet_schema(pool):
    """Parquet 文件包含所有 15 列。"""
    pool.add(_make_entry("a", metrics={"sharpe": 1.0, "ic_mean": 0.05}))
    df = pd.read_parquet(pool.base_dir / "trajectories.parquet")
    expected_cols = {
        "entry_id", "round_idx", "operation", "parent_ids",
        "decision", "duration_ms", "timestamp", "factor_name", "summary",
        "ic_mean", "rank_ic_mean", "sharpe", "arr", "mdd", "calmar",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 1
    assert df.iloc[0]["sharpe"] == 1.0


def test_concurrent_writes(tmp_path):
    """并发写入安全 (Lock)。"""
    pool = TrajectoryPool(tmp_path)

    def writer(eid: str):
        pool.add(_make_entry(eid))

    threads = [threading.Thread(target=writer, args=(f"e{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pool.size == 20
    df = pd.read_parquet(tmp_path / "trajectories.parquet")
    assert len(df) == 20


# ============================================================================
# 7. 边界 (2)
# ============================================================================

def test_pool_get_missing_raises(pool):
    """get 不存在 ID 抛 KeyError。"""
    with pytest.raises(KeyError, match="不存在"):
        pool.get("nonexistent")


def test_selector_invalid_strategy():
    """未知 strategy 抛 ValueError。"""
    with pytest.raises(ValueError, match="未知 strategy"):
        ParentSelector(strategy="invalid")


def test_selector_empty_pool():
    """空 pool 返回空列表。"""
    sel = ParentSelector(strategy="best")
    assert sel.select([], n=3) == []


def test_selector_filters_failed_feedback():
    """feedback.decision=False 不参与选择。"""
    sel = ParentSelector(strategy="best", metric="sharpe")
    pool = [
        _make_entry("pass", metrics={"sharpe": 1.0}, decision=True),
        _make_entry("fail", metrics={"sharpe": 100.0}, decision=False),
    ]
    result = sel.select(pool, n=1)
    assert [e.entry_id for e in result] == ["pass"]


# ============================================================================
# M1: Operation enum
# ============================================================================

from QuantNodes.core.trajectory import Operation


class TestOperationEnum:
    def test_three_values(self):
        assert Operation.ORIGINAL.value == "original"
        assert Operation.MUTATION.value == "mutation"
        assert Operation.CROSSOVER.value == "crossover"

    def test_str_enum(self):
        """str Enum → 直接当字符串用。"""
        assert Operation.ORIGINAL == "original"
        assert isinstance(Operation.ORIGINAL, str)

    def test_from_string(self):
        assert Operation("original") is Operation.ORIGINAL
        assert Operation("mutation") is Operation.MUTATION
        assert Operation("crossover") is Operation.CROSSOVER

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Operation("unknown_op")

    @pytest.mark.parametrize("op,expected_value", [
        (Operation.ORIGINAL, "original"),
        (Operation.MUTATION, "mutation"),
        (Operation.CROSSOVER, "crossover"),
    ])
    def test_param(self, op, expected_value):
        assert op.value == expected_value

    def test_entry_default_uses_enum(self):
        """TrajectoryEntry default 是 Operation.ORIGINAL。"""
        from QuantNodes.core.trajectory import TrajectoryEntry
        e = TrajectoryEntry()
        # 兼容 str
        assert e.operation.value == "original"
        assert e.operation == Operation.ORIGINAL

    def test_entry_accepts_enum(self):
        from QuantNodes.core.trajectory import TrajectoryEntry
        e = TrajectoryEntry(operation=Operation.MUTATION)
        assert e.operation == Operation.MUTATION
        assert e.operation == "mutation"  # str 比较

    def test_lineage_dag_colors_use_enum(self):
        """_OPERATION_COLORS 与 Operation enum 同步。"""
        from QuantNodes.core.visualization.lineage_dag import _OPERATION_COLORS
        for op in Operation:
            assert op.value in _OPERATION_COLORS
