"""trajectory 模块边界条件测试 (25 tests)。

聚焦:
    - TrajectoryEntry: 序列化往返 (JSON / Parquet)、缺字段
    - TrajectoryPool: 并发 add、reset、filter、best、random
    - lineage: 环检测、多 parent、深度限制
    - 并发安全: 50 threads add 验证无丢失
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import FactorFeedback, FeedbackChannel
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool
from QuantNodes.core.trajectory.lineage import children_of, descendants, lineage


# ── 辅助函数 ──────────────────────────────────────────────────

def _make_entry(
    entry_id: str,
    round_idx: int = 0,
    op: str = "original",
    parent_ids: list[str] | None = None,
    decision: bool = True,
    sharpe: float = 0.5,
    factor_name: str = "f_test",
    config: dict | None = None,
) -> TrajectoryEntry:
    fb = FactorFeedback(
        factor_id=entry_id, factor_name=factor_name,
        decision=decision, summary=f"sharpe={sharpe}",
        channels={
            FeedbackChannel.VALUE: FactorFeedback(
                factor_id=entry_id, factor_name=factor_name,
                decision=decision, summary="ok",
            ).channels.get(FeedbackChannel.VALUE) or
            # fallback
            __import__("QuantNodes.core.feedback", fromlist=["ChannelFeedback"]).ChannelFeedback(
                channel=FeedbackChannel.VALUE, passed=decision,
                detail="ok", score=1.0,
            ),
        } if decision else {},
    ) if decision else None
    return TrajectoryEntry(
        entry_id=entry_id,
        round_idx=round_idx,
        operation=op,
        parent_ids=parent_ids or [],
        config_snapshot=config or {"factor": {"name": factor_name, "expression": "close"}},
        feedback=fb,
        metrics={"sharpe": sharpe, "ic_mean": 0.04},
    )


# ============================================================================
# 1. TrajectoryEntry 序列化 (8 tests)
# ============================================================================

class TestTrajectoryEntry:
    def test_default_factory_id(self):
        """不传 entry_id 自动生成 UUID。"""
        e1 = TrajectoryEntry()
        e2 = TrajectoryEntry()
        assert e1.entry_id != e2.entry_id
        assert len(e1.entry_id) >= 32  # UUID 长度

    def test_parquet_row_flat(self):
        """to_parquet_row 输出固定列。"""
        e = _make_entry("e1", sharpe=1.5)
        row = e.to_parquet_row()
        assert "ic_mean" in row
        assert "sharpe" in row
        assert row["sharpe"] == 1.5
        assert row["entry_id"] == "e1"
        assert row["decision"] is True

    def test_parquet_row_no_feedback(self):
        """无 feedback 时不崩。"""
        e = TrajectoryEntry(entry_id="e1", feedback=None)
        row = e.to_parquet_row()
        assert row["decision"] is False
        assert row["duration_ms"] == 0.0
        assert row["factor_name"] == ""

    def test_json_dict_roundtrip(self):
        """JSON 序列化往返。"""
        e = _make_entry("e1", sharpe=1.5, factor_name="alpha_test")
        d = e.to_json_dict()
        json_str = json.dumps(d)
        loaded = TrajectoryEntry.from_json_dict(json.loads(json_str))
        assert loaded.entry_id == "e1"
        assert loaded.round_idx == 0
        assert loaded.metrics["sharpe"] == 1.5
        assert loaded.feedback.factor_name == "alpha_test"
        assert loaded.feedback.decision is True

    def test_json_dict_with_timestamp(self):
        """datetime 字段被 isoformat 化, 还原后相等。"""
        ts = datetime(2025, 6, 11, 10, 30, 0)
        e = TrajectoryEntry(entry_id="e1", timestamp=ts)
        d = e.to_json_dict()
        assert d["timestamp"] == "2025-06-11T10:30:00"
        e2 = TrajectoryEntry.from_json_dict(d)
        assert e2.timestamp == ts

    def test_json_dict_with_pd_timestamp(self):
        """pd.Timestamp 在 config_snapshot 中被 jsonify。"""
        ts = pd.Timestamp("2025-06-11")
        e = TrajectoryEntry(entry_id="e1", config_snapshot={"date": ts})
        d = e.to_json_dict()
        assert d["config_snapshot"]["date"] == "2025-06-11T00:00:00"

    def test_no_feedback_roundtrip(self):
        """无 feedback 也能序列化/反序列化。"""
        e = TrajectoryEntry(entry_id="e1", feedback=None)
        e2 = TrajectoryEntry.from_json_dict(e.to_json_dict())
        assert e2.feedback is None
        assert e2.entry_id == "e1"

    def test_config_with_ndarray(self):
        """config_snapshot 含 ndarray 时 jsonify 转 str。"""
        arr = np.array([1, 2, 3])
        e = TrajectoryEntry(entry_id="e1", config_snapshot={"arr": arr})
        d = e.to_json_dict()
        # 数组转字符串
        assert isinstance(d["config_snapshot"]["arr"], str)


# ============================================================================
# 2. TrajectoryPool CRUD + 过滤 (10 tests)
# ============================================================================

class TestTrajectoryPool:
    def test_empty_pool_size(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 0
        assert len(list(pool)) == 0
        assert pool.all() == []

    def test_add_and_get(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        e = _make_entry("e1")
        pool.add(e)
        assert pool.size == 1
        assert pool.get("e1").entry_id == "e1"

    def test_get_missing_raises(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        with pytest.raises(KeyError, match="entry_id 不存在"):
            pool.get("missing")

    def test_persist_to_disk(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        e = _make_entry("e1", sharpe=1.5)
        pool.add(e)
        # parquet 写入
        assert (tmp_path / "trajectories.parquet").exists()
        # JSON 写入 (entries/ 子目录)
        assert (tmp_path / "entries" / "e1.json").exists()

    def test_reload_from_disk(self, tmp_path: Path):
        pool1 = TrajectoryPool(tmp_path)
        pool1.add(_make_entry("e1", sharpe=1.0))
        pool1.add(_make_entry("e2", sharpe=2.0))
        # 重新加载
        pool2 = TrajectoryPool(tmp_path)
        assert pool2.size == 2
        assert pool2.get("e1").metrics["sharpe"] == 1.0
        assert pool2.get("e2").metrics["sharpe"] == 2.0

    def test_reset_clears_disk(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1"))
        pool.reset()
        assert pool.size == 0
        assert not (tmp_path / "trajectories.parquet").exists()
        assert not (tmp_path / "e1.json").exists()

    def test_by_round_filter(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", round_idx=0))
        pool.add(_make_entry("e2", round_idx=1))
        pool.add(_make_entry("e3", round_idx=1))
        assert len(pool.by_round(0)) == 1
        assert len(pool.by_round(1)) == 2
        assert len(pool.by_round(99)) == 0

    def test_by_operation_filter(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", op="original"))
        pool.add(_make_entry("e2", op="mutation"))
        pool.add(_make_entry("e3", op="crossover"))
        assert len(pool.by_operation("mutation")) == 1
        assert len(pool.by_operation("crossover")) == 1

    def test_filter_by_decision(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", decision=True, sharpe=0.5))
        # 构造 decision=False 但 feedback 不为 None 的 entry
        e2 = TrajectoryEntry(
            entry_id="e2",
            feedback=FactorFeedback(factor_id="e2", factor_name="f2", decision=False),
        )
        pool.add(e2)
        passed = pool.filter(decision=True)
        rejected = pool.filter(decision=False)
        all_entries = pool.filter(decision=None)
        assert len(passed) == 1
        assert passed[0].entry_id == "e1"
        assert len(rejected) == 1
        assert rejected[0].entry_id == "e2"
        assert len(all_entries) == 2

    def test_filter_decision_no_feedback(self, tmp_path: Path):
        """feedback=None 的 entry 在 filter(True/False) 中不出现。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", decision=True))
        e2 = TrajectoryEntry(entry_id="e2", feedback=None)  # 无 feedback
        pool.add(e2)
        passed = pool.filter(decision=True)
        assert len(passed) == 1  # 只有 e1
        assert passed[0].entry_id == "e1"

    def test_best_top_n(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", sharpe=0.5))
        pool.add(_make_entry("e2", sharpe=2.0))
        pool.add(_make_entry("e3", sharpe=1.0))
        top = pool.best(top_n=2, metric="sharpe")
        assert len(top) == 2
        assert top[0].entry_id == "e2"  # sharpe=2.0
        assert top[1].entry_id == "e3"  # sharpe=1.0

    def test_random_n_reproducible(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        for i in range(10):
            pool.add(_make_entry(f"e{i}", sharpe=float(i)))
        r1 = pool.random(3, seed=42)
        r2 = pool.random(3, seed=42)
        ids1 = sorted([e.entry_id for e in r1])
        ids2 = sorted([e.entry_id for e in r2])
        assert ids1 == ids2

    def test_random_n_larger_than_pool(self, tmp_path: Path):
        """random n > pool size 返回全部。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1"))
        pool.add(_make_entry("e2"))
        r = pool.random(10, seed=42)
        assert len(r) == 2

    def test_random_empty_pool(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        assert pool.random(5, seed=42) == []

    def test_iter_is_chronological(self, tmp_path: Path):
        """__iter__ 按 timestamp 排序。"""
        pool = TrajectoryPool(tmp_path)
        e1 = TrajectoryEntry(
            entry_id="e1", timestamp=datetime(2025, 1, 1),
            feedback=FactorFeedback(factor_id="e1", factor_name="f1", decision=True),
        )
        e2 = TrajectoryEntry(
            entry_id="e2", timestamp=datetime(2025, 1, 2),
            feedback=FactorFeedback(factor_id="e2", factor_name="f2", decision=True),
        )
        # 倒序 add
        pool.add(e2)
        pool.add(e1)
        ordered = list(pool)
        assert ordered[0].entry_id == "e1"
        assert ordered[1].entry_id == "e2"


# ============================================================================
# 3. TrajectoryPool 并发安全 (3 tests)
# ============================================================================

class TestConcurrentAdd:
    def test_concurrent_add_50_threads(self, tmp_path: Path):
        """50 threads 同时 add, 验证无丢失。"""
        pool = TrajectoryPool(tmp_path)

        def add_one(i):
            pool.add(_make_entry(f"e{i:03d}", sharpe=float(i)))

        threads = [threading.Thread(target=add_one, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert pool.size == 50

    def test_concurrent_add_with_duplicate_id(self, tmp_path: Path):
        """并发 add 相同 ID, 后者覆盖前者。"""
        pool = TrajectoryPool(tmp_path)
        results = []

        def add_with_sharpe(s):
            pool.add(_make_entry("e1", sharpe=s))
            results.append(s)

        threads = [threading.Thread(target=add_with_sharpe, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 最终 sharpe 是某次写入的值
        assert pool.size == 1
        assert pool.get("e1").metrics["sharpe"] in {float(i) for i in range(10)}


# ============================================================================
# 4. Lineage (5 tests)
# ============================================================================

class TestLineage:
    def test_children_of_empty(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        assert children_of(pool.all(), "missing") == []

    def test_children_of_multiple(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("p", round_idx=0))
        pool.add(_make_entry("c1", round_idx=1, parent_ids=["p"]))
        pool.add(_make_entry("c2", round_idx=1, parent_ids=["p"]))
        pool.add(_make_entry("gc", round_idx=2, parent_ids=["c1"]))
        kids = children_of(pool.all(), "p")
        ids = {k.entry_id for k in kids}
        assert ids == {"c1", "c2"}

    def test_lineage_chain(self, tmp_path: Path):
        """3 代谱系: p → c → gc。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("p", round_idx=0))
        pool.add(_make_entry("c", round_idx=1, parent_ids=["p"]))
        pool.add(_make_entry("gc", round_idx=2, parent_ids=["c"]))
        chain = pool.lineage("gc")
        ids = [e.entry_id for e in chain]
        assert ids == ["p", "c", "gc"]  # 从老到新

    def test_lineage_cycle_prevention(self, tmp_path: Path):
        """人为构造环 (a→b→a), 不会无限循环。"""
        # 用直接 dict 传给 lineage (避免持久化)
        a = TrajectoryEntry(entry_id="a", parent_ids=["b"])
        b = TrajectoryEntry(entry_id="b", parent_ids=["a"])
        entries = {"a": a, "b": b}
        chain = lineage(entries, "a")
        # 防环: 同一 entry_id 不重复访问
        ids = [e.entry_id for e in chain]
        assert len(ids) == 2  # 不应该无限循环
        assert set(ids) == {"a", "b"}

    def test_descendants_with_max_depth(self, tmp_path: Path):
        """max_depth 限制深度。"""
        # 用直接 dict 传给 descendants (更精确, 不依赖 pool.all() 的 dict 视图)
        entries = {
            "p": TrajectoryEntry(entry_id="p", round_idx=0),
            "c1": TrajectoryEntry(entry_id="c1", round_idx=1, parent_ids=["p"]),
            "gc1": TrajectoryEntry(entry_id="gc1", round_idx=2, parent_ids=["c1"]),
            "ggc1": TrajectoryEntry(entry_id="ggc1", round_idx=3, parent_ids=["gc1"]),
        }
        # max_depth=1: 只到 c1
        d1 = descendants(entries, "p", max_depth=1)
        assert {e.entry_id for e in d1} == {"c1"}
        # max_depth=2: 到 gc1
        d2 = descendants(entries, "p", max_depth=2)
        assert {e.entry_id for e in d2} == {"c1", "gc1"}
        # max_depth=None: 全展开
        dn = descendants(entries, "p", max_depth=None)
        assert {e.entry_id for e in dn} == {"c1", "gc1", "ggc1"}

    def test_descendants_missing_entry(self, tmp_path: Path):
        """entry_id 不存在返回空。"""
        pool = TrajectoryPool(tmp_path)
        assert descendants(pool, "missing") == []
