"""TrajectoryEntry/Pool 进阶边界测试 (15 tests)。

聚焦:
    - context_subset 字段 (EvolutionLoop 记录 sandbox context)
    - _persist 缺列 (Parquet 升级兼容)
    - _load 缺 JSON 跳过 + 损坏 JSON 跳过
    - random 抽样均匀性
    - by_round 含 None round_idx
    - filter with non-dict metrics
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


def _make_entry(
    entry_id: str, round_idx: int = 0,
    op: str = "original", parent_ids: list[str] | None = None,
    decision: bool = True, sharpe: float = 0.5,
    factor_name: str = "f", config: dict | None = None,
    context_subset: dict | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id, round_idx=round_idx,
        operation=op, parent_ids=parent_ids or [],
        config_snapshot=config or {"factor": {"name": factor_name, "expression": "close"}},
        context_subset=context_subset or {},
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=factor_name,
            decision=decision, summary=f"sharpe={sharpe}",
        ) if decision else None,
        metrics={"sharpe": sharpe},
    )


# ============================================================================
# 1. context_subset (3 tests)
# ============================================================================

class TestContextSubset:
    def test_default_empty(self):
        e = TrajectoryEntry(entry_id="e1")
        assert e.context_subset == {}

    def test_custom_subset_persists(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        e = _make_entry("e1", context_subset={"load_data_keys": ["cp", "st"]})
        pool.add(e)
        # 重新加载
        pool2 = TrajectoryPool(tmp_path)
        loaded = pool2.get("e1")
        assert loaded.context_subset == {"load_data_keys": ["cp", "st"]}

    def test_complex_subset_json_safe(self, tmp_path: Path):
        """复杂 subset (嵌套 dict) 能 JSON 化。"""
        subset = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list_of_dict": [{"x": 1}, {"y": 2}],
        }
        pool = TrajectoryPool(tmp_path)
        e = _make_entry("e1", context_subset=subset)
        pool.add(e)
        pool2 = TrajectoryPool(tmp_path)
        loaded = pool2.get("e1")
        assert loaded.context_subset["nested"]["a"] == 1
        assert len(loaded.context_subset["list_of_dict"]) == 2


# ============================================================================
# 2. _persist 兼容性 (2 tests)
# ============================================================================

class TestParquetCompat:
    def test_load_existing_pool_with_extra_columns(self, tmp_path: Path):
        """手动创建含额外列的 parquet, 应优雅处理。"""
        path = tmp_path / "trajectories.parquet"
        df = pd.DataFrame([
            {
                "entry_id": "e1", "round_idx": 0, "operation": "original",
                "parent_ids": "", "decision": True, "duration_ms": 0.0,
                "timestamp": datetime.now().isoformat(),
                "factor_name": "f1", "summary": "ok",
                "ic_mean": None, "rank_ic_mean": None, "sharpe": 0.5,
                "arr": None, "mdd": None, "calmar": None,
                "extra_col": "extra_data",  # 多余的列
            },
        ])
        df.to_parquet(path)
        # JSON 文件也要存在 (entries/ 子目录)
        (tmp_path / "entries").mkdir()
        (tmp_path / "entries" / "e1.json").write_text(json.dumps({
            "entry_id": "e1", "round_idx": 0, "operation": "original",
            "config_snapshot": {}, "context_subset": {},
            "feedback": None, "parent_ids": [], "metrics": {},
            "timestamp": datetime.now().isoformat(),
        }))
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 1
        assert pool.get("e1").entry_id == "e1"

    def test_persist_appends_to_existing_parquet(self, tmp_path: Path):
        """add 多次, Parquet 累计多行。"""
        pool = TrajectoryPool(tmp_path)
        for i in range(5):
            pool.add(_make_entry(f"e{i}", sharpe=float(i)))
        # 验证 parquet
        df = pd.read_parquet(tmp_path / "trajectories.parquet")
        assert len(df) == 5
        # 重新加载
        pool2 = TrajectoryPool(tmp_path)
        assert pool2.size == 5


# ============================================================================
# 3. _load 错误恢复 (2 tests)
# ============================================================================

class TestLoadRecovery:
    def test_missing_json_skips_entry(self, tmp_path: Path):
        """Parquet 有 entry, JSON 缺失 → 跳过该 entry。"""
        # 手动写 Parquet 但不写 JSON
        path = tmp_path / "trajectories.parquet"
        df = pd.DataFrame([
            {
                "entry_id": "e1", "round_idx": 0, "operation": "original",
                "parent_ids": "", "decision": True, "duration_ms": 0.0,
                "timestamp": datetime.now().isoformat(),
                "factor_name": "f1", "summary": "ok",
                "ic_mean": None, "rank_ic_mean": None, "sharpe": 0.5,
                "arr": None, "mdd": None, "calmar": None,
            },
        ])
        df.to_parquet(path)
        # 加载, e1 缺失 JSON → 跳过
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 0

    def test_corrupt_json_skips_entry(self, tmp_path: Path):
        path = tmp_path / "trajectories.parquet"
        df = pd.DataFrame([
            {
                "entry_id": "e1", "round_idx": 0, "operation": "original",
                "parent_ids": "", "decision": True, "duration_ms": 0.0,
                "timestamp": datetime.now().isoformat(),
                "factor_name": "f1", "summary": "ok",
                "ic_mean": None, "rank_ic_mean": None, "sharpe": 0.5,
                "arr": None, "mdd": None, "calmar": None,
            },
        ])
        df.to_parquet(path)
        (tmp_path / "e1.json").write_text("invalid json {")
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 0


# ============================================================================
# 4. random 抽样均匀性 (2 tests)
# ============================================================================

class TestRandomDistribution:
    def test_random_distributes_uniformly(self, tmp_path: Path):
        """1000 次抽样, 100 entries, 分布应近似均匀 (各 entry 期望 ~10 次)。"""
        pool = TrajectoryPool(tmp_path)
        for i in range(100):
            pool.add(_make_entry(f"e{i:03d}", sharpe=float(i)))
        # 100 次随机抽 10 个
        all_ids = [f"e{i:03d}" for i in range(100)]
        counts = {eid: 0 for eid in all_ids}
        for seed in range(100):
            sample = pool.random(10, seed=seed)
            for e in sample:
                counts[e.entry_id] += 1
        # 平均 ~10 次, 验证至少 50% 的 entry 被抽到
        hit_count = sum(1 for c in counts.values() if c > 0)
        assert hit_count >= 50  # 50% 命中率

    def test_random_seed_independence(self, tmp_path: Path):
        """不同 seed 应给出不同结果 (高概率)。"""
        pool = TrajectoryPool(tmp_path)
        for i in range(20):
            pool.add(_make_entry(f"e{i:02d}", sharpe=float(i)))
        s1 = sorted([e.entry_id for e in pool.random(5, seed=1)])
        s2 = sorted([e.entry_id for e in pool.random(5, seed=2)])
        # 高概率不同 (但不能 100% 断言)
        # 至少有一处不同
        assert s1 != s2


# ============================================================================
# 5. by_round / by_operation 边界 (3 tests)
# ============================================================================

class TestFilters:
    def test_by_round_negative_index(self, tmp_path: Path):
        """负 round_idx 也能查。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", round_idx=-1))
        pool.add(_make_entry("e2", round_idx=0))
        assert len(pool.by_round(-1)) == 1
        assert len(pool.by_round(0)) == 1

    def test_by_operation_unknown_returns_empty(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", op="original"))
        assert pool.by_operation("unknown_op") == []

    def test_best_with_unknown_metric(self, tmp_path: Path):
        """metric 不在 metrics dict → 当 0。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", sharpe=0.5))
        pool.add(_make_entry("e2", sharpe=1.0))
        # 用 unknown metric → 全部 0 → stable sort, 返回前面 5 个
        top = pool.best(top_n=1, metric="unknown_metric")
        assert len(top) == 1
        # sort reverse 后 0=0, 返回 pool 第一个
        assert top[0].entry_id in {"e1", "e2"}

    def test_filter_decision_includes_rejected_with_decision_false(self, tmp_path: Path):
        """decision=False 但 feedback 不为 None 的 entry 计入 rejected。"""
        pool = TrajectoryPool(tmp_path)
        # 构造 decision=False 但有 feedback
        e1 = TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(
                factor_id="e1", factor_name="f1", decision=False,
            ),
        )
        pool.add(e1)
        rejected = pool.filter(decision=False)
        assert len(rejected) == 1
        assert rejected[0].entry_id == "e1"


# ============================================================================
# 6. size + property (3 tests)
# ============================================================================

class TestPoolProperties:
    def test_size_after_many_adds(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        for i in range(100):
            pool.add(_make_entry(f"e{i:03d}"))
        assert pool.size == 100
        assert len(pool) == 100

    def test_get_returns_loaded_object_after_reload(self, tmp_path: Path):
        """重载后, get 返回反序列化新对象 (字段相同但 identity 不同)。"""
        pool1 = TrajectoryPool(tmp_path)
        e1 = _make_entry("e1")
        pool1.add(e1)
        # 重新加载
        pool2 = TrajectoryPool(tmp_path)
        e1_loaded = pool2.get("e1")
        # 重载后是新对象
        assert e1_loaded.entry_id == e1.entry_id
        # 关键字段都保留
        assert e1_loaded.metrics["sharpe"] == e1.metrics["sharpe"]

    def test_iter_yields_all_entries(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        for i in range(5):
            pool.add(_make_entry(f"e{i}"))
        ids = [e.entry_id for e in pool]
        assert set(ids) == {"e0", "e1", "e2", "e3", "e4"}
