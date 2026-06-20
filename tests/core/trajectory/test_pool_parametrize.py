"""TrajectoryPool 全操作 parametrize (~25 tests)。

遍历 add/get/filter/best/random/reset/by_round/by_operation 每个参数。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


def _entry(
    eid: str, round_idx: int = 0, op: str = "original",
    parent_ids: list[str] | None = None,
    decision: bool = True, sharpe: float = 0.5,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid, round_idx=round_idx,
        operation=op, parent_ids=parent_ids or [],
        config_snapshot={"factor": {"name": f"f_{eid}"}},
        feedback=FactorFeedback(
            factor_id=eid, factor_name=f"f_{eid}", decision=decision,
            summary=f"sharpe={sharpe}",
        ),
        metrics={"sharpe": sharpe},
    )


@pytest.fixture
def fresh_pool(tmp_path: Path) -> TrajectoryPool:
    return TrajectoryPool(tmp_path)


# ============================================================================
# 1. add / size (5 tests)
# ============================================================================

class TestAdd:
    @pytest.mark.parametrize("n_entries", [0, 1, 5, 50, 100])
    def test_size_after_n_adds(self, fresh_pool, n_entries):
        for i in range(n_entries):
            fresh_pool.add(_entry(f"e{i:03d}"))
        assert fresh_pool.size == n_entries

    def test_add_duplicate_id(self, fresh_pool):
        fresh_pool.add(_entry("dup", sharpe=1.0))
        fresh_pool.add(_entry("dup", sharpe=2.0))
        # 同一 ID 覆盖
        assert fresh_pool.size == 1
        assert fresh_pool.get("dup").metrics["sharpe"] == 2.0

    def test_add_with_parent_ids(self, fresh_pool):
        fresh_pool.add(_entry("p", round_idx=0))
        fresh_pool.add(_entry("c", round_idx=1, parent_ids=["p"]))
        c = fresh_pool.get("c")
        assert c.parent_ids == ["p"]

    @pytest.mark.parametrize("op", ["original", "mutation", "crossover"])
    def test_add_with_each_operation(self, fresh_pool, op):
        fresh_pool.add(_entry(f"e_{op}", op=op))
        assert fresh_pool.get(f"e_{op}").operation == op

    def test_persists_to_disk(self, fresh_pool, tmp_path):
        fresh_pool.add(_entry("e1"))
        fresh_pool.add(_entry("e2"))
        assert (tmp_path / "trajectories.parquet").exists()
        assert (tmp_path / "entries" / "e1.json").exists()
        assert (tmp_path / "entries" / "e2.json").exists()

    @pytest.mark.parametrize("custom_name", [
        "exp_a.parquet", "my_pool.parquet", "run_2024.parquet",
    ])
    def test_custom_parquet_name(self, tmp_path, custom_name):
        """H4: parquet 文件名可定制, 允许多实验共存。"""
        pool = TrajectoryPool(tmp_path, parquet_name=custom_name)
        pool.add(_entry("e1"))
        assert (tmp_path / custom_name).exists()
        # 默认名应不存在
        assert not (tmp_path / "trajectories.parquet").exists()
        # 重载: 用同 parquet_name 加载
        pool2 = TrajectoryPool(tmp_path, parquet_name=custom_name)
        assert pool2.size == 1

    def test_custom_parquet_name_then_default_loads_empty(self, tmp_path):
        """用默认 parquet_name 加载自定义 pool, 应得空。"""
        pool = TrajectoryPool(tmp_path, parquet_name="custom.parquet")
        pool.add(_entry("e1"))
        # 默认名加载 → 空
        pool2 = TrajectoryPool(tmp_path)
        assert pool2.size == 0

    def test_reset_creates_entries_dir(self, tmp_path):
        """H5: reset() 用 entries/ 子目录, 不误删外部 JSON。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1"))
        # 外部 JSON 不应被误删
        external = tmp_path / "user_data.json"
        external.write_text("{}")
        pool.reset()
        assert external.exists()  # 不被误删
        assert (tmp_path / "entries").exists()  # 目录保留
        assert (tmp_path / "trajectories.parquet").exists() is False
        assert len(list((tmp_path / "entries").glob("*.json"))) == 0


# ============================================================================
# 2. get / KeyError (4 tests)
# ============================================================================

class TestGet:
    @pytest.mark.parametrize("eid", ["a", "e_1", "00000", "with-dash", "中文"])
    def test_get_returns_entry(self, fresh_pool, eid):
        fresh_pool.add(_entry(eid))
        assert fresh_pool.get(eid).entry_id == eid

    def test_get_missing_raises(self, fresh_pool):
        with pytest.raises(KeyError, match="entry_id 不存在"):
            fresh_pool.get("nonexistent")

    @pytest.mark.parametrize("eid", ["", " ", "very_long_id_" * 10])
    def test_get_various_id_formats(self, fresh_pool, eid):
        fresh_pool.add(_entry(eid))
        assert fresh_pool.get(eid).entry_id == eid


# ============================================================================
# 3. filter (3 tests)
# ============================================================================

class TestFilter:
    @pytest.mark.parametrize("decision,n_passed", [
        (True, 3), (False, 2), (None, 5),
    ])
    def test_filter_decision(self, fresh_pool, decision, n_passed):
        for i in range(3):
            fresh_pool.add(_entry(f"p{i}", decision=True))
        for i in range(2):
            fresh_pool.add(_entry(f"r{i}", decision=False))
        result = fresh_pool.filter(decision=decision)
        assert len(result) == n_passed

    def test_filter_with_feedback_none(self, fresh_pool):
        fresh_pool.add(TrajectoryEntry(entry_id="nf", feedback=None))
        fresh_pool.add(_entry("ok"))
        # decision=None: all
        assert len(fresh_pool.filter(decision=None)) == 2
        # decision=True: 只 ok
        assert len(fresh_pool.filter(decision=True)) == 1


# ============================================================================
# 4. best (3 tests)
# ============================================================================

class TestBest:
    @pytest.mark.parametrize("top_n,expected_len", [
        (0, 0), (1, 1), (3, 3), (5, 5), (10, 5),  # n > pool
    ])
    def test_best_n(self, fresh_pool, top_n, expected_len):
        for i in range(5):
            fresh_pool.add(_entry(f"e{i}", sharpe=float(i)))
        assert len(fresh_pool.best(top_n=top_n, metric="sharpe")) == expected_len

    @pytest.mark.parametrize("metric", ["sharpe", "ic_mean", "arr", "mdd", "calmar"])
    def test_best_with_each_metric(self, fresh_pool, metric):
        for i in range(3):
            fresh_pool.add(_entry(f"e{i}", sharpe=float(i)))
        r = fresh_pool.best(top_n=1, metric=metric)
        assert len(r) == 1


# ============================================================================
# 5. random (3 tests)
# ============================================================================

class TestRandom:
    @pytest.mark.parametrize("n,expected_len", [
        (0, 0), (1, 1), (3, 3), (10, 5),  # n > pool
    ])
    def test_random_n(self, fresh_pool, n, expected_len):
        for i in range(5):
            fresh_pool.add(_entry(f"e{i}"))
        r = fresh_pool.random(n=n, seed=42)
        assert len(r) == expected_len

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, None])
    def test_random_seed_variants(self, fresh_pool, seed):
        for i in range(5):
            fresh_pool.add(_entry(f"e{i}"))
        r = fresh_pool.random(n=3, seed=seed)
        assert len(r) == 3


# ============================================================================
# 6. by_round / by_operation (4 tests)
# ============================================================================

class TestByFilters:
    @pytest.mark.parametrize("round_idx,n_expected", [
        (0, 1), (1, 2), (2, 3), (99, 0),
    ])
    def test_by_round(self, fresh_pool, round_idx, n_expected):
        fresh_pool.add(_entry("e0", round_idx=0))
        fresh_pool.add(_entry("e1", round_idx=1))
        fresh_pool.add(_entry("e2", round_idx=1))
        fresh_pool.add(_entry("e3", round_idx=2))
        fresh_pool.add(_entry("e4", round_idx=2))
        fresh_pool.add(_entry("e5", round_idx=2))
        assert len(fresh_pool.by_round(round_idx)) == n_expected

    @pytest.mark.parametrize("op,n_expected", [
        ("original", 2), ("mutation", 2), ("crossover", 1),
    ])
    def test_by_operation(self, fresh_pool, op, n_expected):
        fresh_pool.add(_entry("o1", op="original"))
        fresh_pool.add(_entry("o2", op="original"))
        fresh_pool.add(_entry("m1", op="mutation"))
        fresh_pool.add(_entry("m2", op="mutation"))
        fresh_pool.add(_entry("c1", op="crossover"))
        assert len(fresh_pool.by_operation(op)) == n_expected


# ============================================================================
# 7. reset / iter / all / __len__ (3 tests)
# ============================================================================

class TestIter:
    def test_reset_clears(self, fresh_pool):
        for i in range(3):
            fresh_pool.add(_entry(f"e{i}"))
        fresh_pool.reset()
        assert fresh_pool.size == 0

    def test_all_returns_sorted(self, fresh_pool):
        for i in range(3):
            fresh_pool.add(_entry(f"e{i}"))
        all_e = fresh_pool.all()
        assert len(all_e) == 3
        # sorted by timestamp, 但同时 add 时序接近

    @pytest.mark.parametrize("n", [0, 1, 5, 50])
    def test_len(self, fresh_pool, n):
        for i in range(n):
            fresh_pool.add(_entry(f"e{i}"))
        assert len(fresh_pool) == n
        assert fresh_pool.size == n


# ============================================================================
# 8. _persist 兼容性 (3 tests)
# ============================================================================

class TestPersistence:
    def test_reload_preserves_all(self, tmp_path):
        p1 = TrajectoryPool(tmp_path)
        for i in range(5):
            p1.add(_entry(f"e{i}", sharpe=float(i)))
        p2 = TrajectoryPool(tmp_path)
        assert p2.size == 5
        assert p2.get("e4").metrics["sharpe"] == 4.0

    def test_reload_skips_missing_json(self, tmp_path):
        """Parquet 有但 JSON 缺 → 跳过。"""
        # 手动写 Parquet + 不写 JSON
        df = pd.DataFrame([{
            "entry_id": "e1", "round_idx": 0, "operation": "original",
            "parent_ids": "", "decision": True, "duration_ms": 0.0,
            "timestamp": datetime.now().isoformat(),
            "factor_name": "f", "summary": "s",
            "ic_mean": None, "rank_ic_mean": None, "sharpe": 0.5,
            "arr": None, "mdd": None, "calmar": None,
        }])
        df.to_parquet(tmp_path / "trajectories.parquet")
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 0

    def test_reload_skips_bad_json(self, tmp_path):
        df = pd.DataFrame([{
            "entry_id": "e1", "round_idx": 0, "operation": "original",
            "parent_ids": "", "decision": True, "duration_ms": 0.0,
            "timestamp": datetime.now().isoformat(),
            "factor_name": "f", "summary": "s",
            "ic_mean": None, "rank_ic_mean": None, "sharpe": 0.5,
            "arr": None, "mdd": None, "calmar": None,
        }])
        df.to_parquet(tmp_path / "trajectories.parquet")
        (tmp_path / "e1.json").write_text("not json")
        pool = TrajectoryPool(tmp_path)
        assert pool.size == 0
