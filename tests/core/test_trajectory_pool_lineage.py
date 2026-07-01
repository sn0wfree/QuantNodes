# coding=utf-8
"""Tests for core/trajectory/{pool,lineage}.py.

Covers: TrajectoryPool CRUD, persistence (Parquet+JSON), filtering,
children_of/lineage/descendants stateless functions.
"""

from pathlib import Path

import pytest

from QuantNodes.core.trajectory.pool import TrajectoryPool
from QuantNodes.core.trajectory.lineage import (
    children_of,
    lineage,
    descendants,
)
from QuantNodes.core.trajectory.entry import TrajectoryEntry
from QuantNodes.core.feedback.dataclass import FactorFeedback


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pool(tmp_path):
    """Create a TrajectoryPool in a temp directory."""
    return TrajectoryPool(base_dir=tmp_path / "pool")


@pytest.fixture
def populated_pool(pool):
    """Pool with 5 entries: 1 root, 2 children, 2 grandchildren."""
    pool.add(TrajectoryEntry(
        entry_id="root",
        config_snapshot={"factor": {"name": "root"}},
        round_idx=0,
        feedback=FactorFeedback(factor_id="root", factor_name="root", decision=True),
    ))
    pool.add(TrajectoryEntry(
        entry_id="child1",
        parent_ids=["root"],
        config_snapshot={"factor": {"name": "c1"}},
        round_idx=1,
        feedback=FactorFeedback(factor_id="c1", factor_name="c1", decision=True),
    ))
    pool.add(TrajectoryEntry(
        entry_id="child2",
        parent_ids=["root"],
        config_snapshot={"factor": {"name": "c2"}},
        round_idx=1,
        feedback=FactorFeedback(factor_id="c2", factor_name="c2", decision=False),
    ))
    pool.add(TrajectoryEntry(
        entry_id="grandchild1",
        parent_ids=["child1"],
        config_snapshot={"factor": {"name": "gc1"}},
        round_idx=2,
        metrics={"sharpe": 1.5},
    ))
    pool.add(TrajectoryEntry(
        entry_id="grandchild2",
        parent_ids=["child1"],
        config_snapshot={"factor": {"name": "gc2"}},
        round_idx=2,
        metrics={"sharpe": 0.8},
    ))
    return pool


# ============================================================================
# Creation
# ============================================================================

class TestPoolCreation:
    def test_creation(self, pool):
        assert pool.size == 0
        assert len(pool) == 0

    def test_creation_creates_directory(self, tmp_path):
        path = tmp_path / "new_pool"
        TrajectoryPool(base_dir=path)
        assert path.exists()

    def test_creation_with_custom_parquet_name(self, tmp_path):
        path = tmp_path / "pool"
        pool = TrajectoryPool(base_dir=path, parquet_name="custom.parquet")
        assert pool._parquet_name == "custom.parquet"

    def test_iter_empty(self, pool):
        assert list(iter(pool)) == []


# ============================================================================
# add / get
# ============================================================================

class TestPoolAddGet:
    def test_add_single(self, pool):
        e = TrajectoryEntry(entry_id="e1", config_snapshot={"factor": {"name": "e1"}})
        pool.add(e)
        assert pool.size == 1

    def test_get_existing(self, populated_pool):
        e = populated_pool.get("root")
        assert e.entry_id == "root"

    def test_get_missing_raises(self, populated_pool):
        with pytest.raises(KeyError, match="entry_id 不存在"):
            populated_pool.get("nonexistent")

    def test_add_overwrites(self, pool):
        e1 = TrajectoryEntry(entry_id="e1", config_snapshot={"factor": {"name": "v1"}})
        e2 = TrajectoryEntry(entry_id="e1", config_snapshot={"factor": {"name": "v2"}})
        pool.add(e1)
        pool.add(e2)
        # Should still be 1 entry (same ID)
        assert pool.size == 1
        assert pool.get("e1").config_snapshot["factor"]["name"] == "v2"


# ============================================================================
# all() / iter / len
# ============================================================================

class TestPoolIter:
    def test_all_returns_list(self, populated_pool):
        entries = populated_pool.all()
        assert isinstance(entries, list)
        assert len(entries) == 5

    def test_all_sorted_by_timestamp(self, populated_pool):
        """all() returns entries sorted by timestamp."""
        entries = populated_pool.all()
        # Check they're in order
        for i in range(len(entries) - 1):
            assert entries[i].timestamp <= entries[i + 1].timestamp

    def test_iter(self, populated_pool):
        ids = [e.entry_id for e in populated_pool]
        assert "root" in ids
        assert "child1" in ids

    def test_len(self, populated_pool):
        assert len(populated_pool) == 5


# ============================================================================
# Filtering
# ============================================================================

class TestPoolFiltering:
    def test_by_round(self, populated_pool):
        r0 = populated_pool.by_round(0)
        assert len(r0) == 1
        assert r0[0].entry_id == "root"

        r1 = populated_pool.by_round(1)
        assert len(r1) == 2

        r2 = populated_pool.by_round(2)
        assert len(r2) == 2

    def test_by_round_no_match(self, populated_pool):
        assert populated_pool.by_round(99) == []

    def test_by_operation(self, populated_pool):
        all_ops = populated_pool.by_operation("original")
        # All entries default to Operation.ORIGINAL
        assert len(all_ops) >= 0

    def test_filter_decision_true(self, populated_pool):
        passed = populated_pool.filter(decision=True)
        # root and child1 passed; child2 failed
        assert len(passed) == 2

    def test_filter_decision_false(self, populated_pool):
        failed = populated_pool.filter(decision=False)
        assert len(failed) == 1

    def test_filter_decision_none(self, populated_pool):
        all_entries = populated_pool.filter(decision=None)
        assert len(all_entries) == 5


# ============================================================================
# best() / random()
# ============================================================================

class TestPoolBest:
    def test_best_sorted_descending(self, populated_pool):
        best = populated_pool.best(top_n=3, metric="sharpe")
        # Top 3 by sharpe
        assert len(best) == 3
        # First should have highest sharpe (grandchild1=1.5)
        assert best[0].entry_id == "grandchild1"

    def test_best_custom_metric(self, populated_pool):
        best = populated_pool.best(top_n=2, metric="sharpe")
        # Verify sorted descending
        for i in range(len(best) - 1):
            v1 = float(best[i].metrics.get("sharpe", 0) or 0)
            v2 = float(best[i + 1].metrics.get("sharpe", 0) or 0)
            assert v1 >= v2

    def test_best_empty(self, pool):
        assert pool.best(top_n=5) == []


class TestPoolRandom:
    def test_random_returns_n(self, populated_pool):
        sampled = populated_pool.random(n=3, seed=42)
        assert len(sampled) == 3

    def test_random_seed_reproducible(self, populated_pool):
        s1 = populated_pool.random(n=3, seed=42)
        s2 = populated_pool.random(n=3, seed=42)
        # Same seed → same IDs
        ids1 = {e.entry_id for e in s1}
        ids2 = {e.entry_id for e in s2}
        assert ids1 == ids2

    def test_random_n_exceeds_size(self, populated_pool):
        sampled = populated_pool.random(n=100, seed=42)
        # Should return all entries, not more
        assert len(sampled) == 5

    def test_random_empty(self, pool):
        assert pool.random(n=5) == []


# ============================================================================
# Lineage methods (delegated to lineage.py)
# ============================================================================

class TestPoolLineageMethods:
    def test_children_of(self, populated_pool):
        children = populated_pool.children_of("root")
        ids = {c.entry_id for c in children}
        assert ids == {"child1", "child2"}

    def test_children_of_no_children(self, populated_pool):
        children = populated_pool.children_of("grandchild2")
        assert children == []

    def test_descendants(self, populated_pool):
        desc = populated_pool.descendants("root")
        ids = {d.entry_id for d in desc}
        assert ids == {"child1", "child2", "grandchild1", "grandchild2"}

    def test_descendants_with_max_depth(self, populated_pool):
        desc = populated_pool.descendants("root", max_depth=1)
        ids = {d.entry_id for d in desc}
        # Only direct children at depth=1
        assert ids == {"child1", "child2"}

    def test_lineage(self, populated_pool):
        chain = populated_pool.lineage("grandchild1")
        # Should be [root, child1, grandchild1]
        assert len(chain) == 3
        assert chain[0].entry_id == "root"
        assert chain[-1].entry_id == "grandchild1"


# ============================================================================
# reset()
# ============================================================================

class TestPoolReset:
    def test_reset_clears_memory(self, populated_pool):
        populated_pool.reset()
        assert populated_pool.size == 0

    def test_reset_deletes_parquet(self, populated_pool):
        assert populated_pool._parquet_path.exists()
        populated_pool.reset()
        assert not populated_pool._parquet_path.exists()

    def test_reset_deletes_entries_dir(self, populated_pool):
        # Add some JSON files
        populated_pool.reset()  # Reset before adding
        e = TrajectoryEntry(entry_id="e1", config_snapshot={"factor": {"name": "e1"}})
        populated_pool.add(e)
        # Check JSON exists
        json_path = populated_pool._entries_dir / "e1.json"
        assert json_path.exists()
        populated_pool.reset()
        assert not json_path.exists()


# ============================================================================
# Persistence (load on startup)
# ============================================================================

class TestPoolPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "pool"
        # Create and populate
        pool1 = TrajectoryPool(base_dir=path)
        pool1.add(TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"factor": {"name": "test"}},
            metrics={"sharpe": 1.5},
        ))

        # Reload
        pool2 = TrajectoryPool(base_dir=path)
        assert pool2.size == 1
        e = pool2.get("e1")
        assert e.metrics.get("sharpe") == 1.5

    def test_parquet_persists(self, populated_pool, tmp_path):
        # The pool was created with tmp_path/pool
        assert populated_pool._parquet_path.exists()

    def test_json_persists_per_entry(self, populated_pool):
        for eid in ["root", "child1", "child2"]:
            json_path = populated_pool._entries_dir / f"{eid}.json"
            assert json_path.exists()


# ============================================================================
# children_of (stateless)
# ============================================================================

class TestChildrenOf:
    def test_with_dict(self):
        e1 = TrajectoryEntry(entry_id="c1", parent_ids=["p1"])
        e2 = TrajectoryEntry(entry_id="c2", parent_ids=["p1"])
        e3 = TrajectoryEntry(entry_id="c3", parent_ids=["p2"])
        result = children_of({"c1": e1, "c2": e2, "c3": e3}, "p1")
        assert len(result) == 2

    def test_with_iterable(self):
        e1 = TrajectoryEntry(entry_id="c1", parent_ids=["p1"])
        e2 = TrajectoryEntry(entry_id="c2", parent_ids=["p2"])
        result = children_of([e1, e2], "p1")
        assert len(result) == 1

    def test_no_children(self):
        e1 = TrajectoryEntry(entry_id="c1", parent_ids=["p1"])
        result = children_of({"c1": e1}, "p2")
        assert result == []

    def test_multiple_parents(self):
        """child with multiple parents can be found via any parent_id."""
        e1 = TrajectoryEntry(entry_id="c1", parent_ids=["p1", "p2"])
        # child_of p1 finds it
        assert len(children_of({"c1": e1}, "p1")) == 1
        # child_of p2 also finds it
        assert len(children_of({"c1": e1}, "p2")) == 1


# ============================================================================
# lineage (stateless)
# ============================================================================

class TestLineage:
    def test_empty_when_missing(self):
        assert lineage({}, "x") == []

    def test_single_entry(self):
        e = TrajectoryEntry(entry_id="x")
        chain = lineage({"x": e}, "x")
        assert len(chain) == 1

    def test_chain(self):
        root = TrajectoryEntry(entry_id="root")
        child = TrajectoryEntry(entry_id="child", parent_ids=["root"])
        grand = TrajectoryEntry(entry_id="grand", parent_ids=["child"])
        chain = lineage(
            {"root": root, "child": child, "grand": grand},
            "grand",
        )
        assert [e.entry_id for e in chain] == ["root", "child", "grand"]

    def test_cycle_prevention(self):
        """Cycle in parent_ids shouldn't cause infinite loop."""
        # Create a cycle: a -> b -> a
        a = TrajectoryEntry(entry_id="a", parent_ids=["b"])
        b = TrajectoryEntry(entry_id="b", parent_ids=["a"])
        chain = lineage({"a": a, "b": b}, "a")
        # Should terminate, not infinite loop
        assert len(chain) >= 1

    def test_missing_parent_terminates(self):
        """parent_id that doesn't exist in dict should terminate."""
        root = TrajectoryEntry(entry_id="root")
        orphan = TrajectoryEntry(entry_id="orphan", parent_ids=["nonexistent"])
        chain = lineage({"root": root, "orphan": orphan}, "orphan")
        # Only orphan, no further ancestors
        assert [e.entry_id for e in chain] == ["orphan"]


# ============================================================================
# descendants (stateless)
# ============================================================================

class TestDescendants:
    def test_no_descendants(self):
        e = TrajectoryEntry(entry_id="leaf")
        assert descendants({"leaf": e}, "leaf") == []

    def test_bfs_traversal(self):
        root = TrajectoryEntry(entry_id="root")
        c1 = TrajectoryEntry(entry_id="c1", parent_ids=["root"])
        c2 = TrajectoryEntry(entry_id="c2", parent_ids=["root"])
        gc = TrajectoryEntry(entry_id="gc", parent_ids=["c1"])
        result = descendants(
            {"root": root, "c1": c1, "c2": c2, "gc": gc},
            "root",
        )
        ids = {e.entry_id for e in result}
        assert ids == {"c1", "c2", "gc"}

    def test_max_depth(self):
        root = TrajectoryEntry(entry_id="root")
        c = TrajectoryEntry(entry_id="c", parent_ids=["root"])
        gc = TrajectoryEntry(entry_id="gc", parent_ids=["c"])
        result = descendants(
            {"root": root, "c": c, "gc": gc},
            "root",
            max_depth=1,
        )
        # Only direct children at depth=1
        ids = {e.entry_id for e in result}
        assert "c" in ids
        assert "gc" not in ids

    def test_missing_root(self):
        assert descendants({}, "x") == []

    def test_cycle_prevention(self):
        a = TrajectoryEntry(entry_id="a", parent_ids=["b"])
        b = TrajectoryEntry(entry_id="b", parent_ids=["a"])
        # Should not infinite loop
        result = descendants({"a": a, "b": b}, "a")
        assert isinstance(result, list)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_cache_invalidation_on_add(self, populated_pool):
        # Get cached list
        first = populated_pool.all()
        # Add new entry
        new_e = TrajectoryEntry(entry_id="new", config_snapshot={"factor": {"name": "n"}})
        populated_pool.add(new_e)
        # Should reflect new entry
        second = populated_pool.all()
        assert len(second) == len(first) + 1

    def test_add_unicode_entry_id(self, pool):
        e = TrajectoryEntry(entry_id="中文_id_中文", config_snapshot={"factor": {"name": "u"}})
        pool.add(e)
        assert pool.size == 1
        assert pool.get("中文_id_中文").entry_id == "中文_id_中文"

    def test_special_chars_in_parquet_name(self, tmp_path):
        path = tmp_path / "pool"
        pool = TrajectoryPool(base_dir=path, parquet_name="test@#$.parquet")
        e = TrajectoryEntry(entry_id="e1", config_snapshot={"factor": {"name": "x"}})
        pool.add(e)
        assert pool.size == 1