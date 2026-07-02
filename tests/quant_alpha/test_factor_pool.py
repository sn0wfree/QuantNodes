# coding=utf-8
"""
test_factor_pool.py - FactorPool 单元测试 (v3.0.2)

覆盖:
- 基础 CRUD (add/extend/get/remove/contains/clear)
- dedup / select / filter / summary
- Wiki 双向 (from_wiki / to_wiki) -- 用 MagicMock 模拟 WikiFactorProxy
- JSON 离线持久化
- 线程安全
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.quant_alpha.factor_pool import FactorEntry, FactorPool


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def make_entry():
    """工厂 fixture: 按 formula_id 快速构造 FactorEntry"""

    def _make(
        formula_id: str,
        ir: float = 0.5,
        ic_mean: float = 0.03,
        rank_ic: float = 0.04,
        source_lib: Optional[str] = None,
        source_id: Optional[str] = None,
        formula: Optional[str] = None,
        tags: Optional[List[str]] = None,
        structured: Optional[Dict[str, Any]] = None,
    ) -> FactorEntry:
        # Default: derive source_lib from formula_id prefix ("alpha101-x" -> "alpha101")
        if source_lib is None:
            source_lib = formula_id.split("-", 1)[0] if "-" in formula_id else ""
        if source_id is None:
            source_id = formula_id.split("-", 1)[-1] if "-" in formula_id else formula_id
        return FactorEntry(
            formula_id=formula_id,
            formula=formula or f"formula({formula_id})",
            source_lib=source_lib,
            source_id=source_id,
            ir=ir,
            ic_mean=ic_mean,
            rank_ic=rank_ic,
            tags=tags or [],
            structured=structured,
        )

    return _make


@pytest.fixture
def pool() -> FactorPool:
    return FactorPool(wiki_path="/tmp/wiki_test")


# ======================================================================
# 基础 CRUD
# ======================================================================
class TestFactorEntryBasics:
    def test_construction_minimal(self):
        e = FactorEntry(formula_id="x", formula="f(x)")
        assert e.formula_id == "x"
        assert e.formula == "f(x)"
        assert e.source_lib == ""
        assert e.ir == 0.0
        assert e.ic_mean == 0.0
        assert e.discovered_at != ""
        assert isinstance(e.tags, list)

    def test_construction_full(self, make_entry):
        e = make_entry("alpha101-alpha006", ir=1.2, ic_mean=0.04, tags=["logic-driven"])
        assert e.ir == 1.2
        assert e.ic_mean == 0.04
        assert "logic-driven" in e.tags
        assert e.source_id == "alpha006"

    def test_to_dict_roundtrip(self, make_entry):
        e = make_entry("alpha101-alpha006", ir=0.7, structured={"predicates": []})
        d = e.to_dict()
        restored = FactorEntry.from_dict(d)
        assert restored.formula_id == e.formula_id
        assert restored.ir == e.ir
        assert restored.structured == e.structured

    def test_from_logic_result_helper(self):
        e = FactorEntry.from_logic_result(
            formula_id="alpha101-006",
            formula="close-open",
            source_lib="alpha101",
            source_id="006",
            ir=0.8,
            tags=["x"],
        )
        assert e.formula_id == "alpha101-006"
        assert e.source_id == "006"
        assert e.ir == 0.8


class TestPoolCRUD:
    def test_add_returns_true_when_new(self, pool, make_entry):
        assert pool.add(make_entry("alpha101-alpha006")) is True
        assert len(pool) == 1

    def test_add_returns_false_when_overwrite(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006", ir=0.3))
        assert pool.add(make_entry("alpha101-alpha006", ir=0.9)) is False
        assert pool.get("alpha101-alpha006").ir == 0.9

    def test_extend_counts_new_only(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006"))
        added = pool.extend([
            make_entry("alpha101-alpha006", ir=0.99),
            make_entry("alpha101-alpha007"),
            make_entry("alpha191-alpha001"),
        ])
        assert added == 2
        assert len(pool) == 3
        assert pool.get("alpha101-alpha006").ir == 0.99

    def test_remove_and_contains(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006"))
        assert "alpha101-alpha006" in pool
        assert pool.remove("alpha101-alpha006") is True
        assert "alpha101-alpha006" not in pool
        assert pool.remove("alpha101-alpha006") is False

    def test_keys_values_iteration(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006"))
        pool.add(make_entry("alpha191-alpha001"))
        assert set(pool.keys()) == {"alpha101-alpha006", "alpha191-alpha001"}
        assert len(list(pool)) == 2

    def test_clear(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006"))
        pool.clear()
        assert len(pool) == 0


# ======================================================================
# dedup / select / filter / summary
# ======================================================================
class TestPoolOperations:
    def test_dedup_by_formula_keeps_highest_ir(self, pool, make_entry):
        pool.add(make_entry("alpha101-a1", ir=0.5, formula="f1"))
        pool.add(make_entry("alpha101-a2", ir=0.9, formula="f1"))  # same formula, higher ir
        pool.add(make_entry("alpha101-a3", ir=0.3, formula="f2"))
        removed = pool.dedup(by="formula")
        assert removed == 1
        assert pool.get("alpha101-a2") is not None
        assert pool.get("alpha101-a1") is None

    def test_dedup_by_source_id(self, pool, make_entry):
        pool.add(make_entry("a", source_lib="alpha101", source_id="006", ir=0.3))
        pool.add(make_entry("b", source_lib="alpha101", source_id="006", ir=0.7))
        pool.add(make_entry("c", source_lib="alpha101", source_id="007", ir=0.5))
        removed = pool.dedup(by="source_id")
        assert removed == 1
        assert pool.get("b") is not None

    def test_dedup_by_formula_id_is_noop(self, pool, make_entry):
        pool.add(make_entry("alpha101-a1"))
        assert pool.dedup(by="formula_id") == 0

    def test_dedup_unknown_key_raises(self, pool, make_entry):
        pool.add(make_entry("alpha101-a1"))
        with pytest.raises(ValueError, match="Unknown dedup key"):
            pool.dedup(by="nonsense")

    def test_select_top_n_by_ir(self, pool, make_entry):
        for i, ir in enumerate([0.1, 0.5, 0.9, 0.3, 0.7]):
            pool.add(make_entry(f"alpha101-a{i}", ir=ir))
        top = pool.select(top_n=3, by="ir")
        assert [e.formula_id for e in top] == ["alpha101-a2", "alpha101-a4", "alpha101-a1"]

    def test_select_by_ic_mean(self, pool, make_entry):
        pool.add(make_entry("a", ir=0.9, ic_mean=0.01))
        pool.add(make_entry("b", ir=0.3, ic_mean=0.05))
        top = pool.select(top_n=1, by="ic_mean")
        assert top[0].formula_id == "b"

    def test_select_zero_or_negative_returns_empty(self, pool, make_entry):
        pool.add(make_entry("a"))
        assert pool.select(top_n=0) == []
        assert pool.select(top_n=-1) == []

    def test_filter_by_source_lib_and_min_ir(self, pool, make_entry):
        pool.add(make_entry("alpha101-a", ir=0.9))
        pool.add(make_entry("alpha101-b", ir=0.3))
        pool.add(make_entry("alpha191-c", ir=0.7))
        out = pool.filter(source_lib="alpha101", min_ir=0.5)
        assert [e.formula_id for e in out] == ["alpha101-a"]
        # min_ir 边界: 0.3 < 0.5 仍排除
        out2 = pool.filter(min_ir=0.4)
        assert sorted(e.formula_id for e in out2) == ["alpha101-a", "alpha191-c"]

    def test_filter_by_tags(self, pool, make_entry):
        pool.add(make_entry("a", tags=["logic-driven", "ir=0.9"]))
        pool.add(make_entry("b", tags=["manual"]))
        out = pool.filter(tags=["logic-driven"])
        assert [e.formula_id for e in out] == ["a"]

    def test_summary_empty_pool(self, pool):
        s = pool.summary()
        assert s["n_total"] == 0
        assert s["by_source_lib"] == {}
        assert s["ir_stats"] == {}
        assert s["n_with_wiki"] == 0

    def test_summary_with_entries(self, pool, make_entry):
        pool.add(make_entry("alpha101-a", ir=0.5))
        pool.add(make_entry("alpha101-b", ir=0.9))
        pool.add(make_entry("alpha191-c", ir=0.7))
        s = pool.summary()
        assert s["n_total"] == 3
        assert s["by_source_lib"]["alpha101"] == 2
        assert s["by_source_lib"]["alpha191"] == 1
        assert s["ir_stats"]["max"] == 0.9
        assert s["ir_stats"]["min"] == 0.5
        assert s["ir_stats"]["median"] == pytest.approx(0.7)


# ======================================================================
# Wiki 双向 (用 MagicMock 模拟 WikiFactorProxy)
# ======================================================================
class TestWikiSync:
    def test_from_wiki_loads_existing_logics(self, pool, make_entry):
        # 构造 mock WikiLogic
        logic_a = MagicMock()
        logic_a.name = "alpha101-alpha006"
        logic_a.extracted_formula = "close-open"
        logic_a.source_detail = {"source_lib": "alpha101", "source_id": "006"}
        logic_a.tags = []
        logic_a.created_at = "2026-01-01T00:00:00"
        logic_a.wiki_page_name = "Logic/alpha101-alpha006.md"
        logic_a.structured = MagicMock()
        logic_a.structured.to_dict.return_value = {"predicates": []}
        logic_a.performance_evidence = MagicMock()
        logic_a.performance_evidence.to_dict.return_value = {"best_ir": 0.8}

        proxy = MagicMock()
        proxy.list_logics.return_value = [logic_a]
        n = pool.from_wiki(proxy)
        assert n == 1
        assert pool.get("alpha101-alpha006").ir == 0.8
        assert pool.get("alpha101-alpha006").source_lib == "alpha101"
        assert pool.get("alpha101-alpha006").wiki_path == "Logic/alpha101-alpha006.md"

    def test_from_wiki_handles_list_logics_failure(self, pool):
        proxy = MagicMock()
        proxy.list_logics.side_effect = RuntimeError("disk full")
        assert pool.from_wiki(proxy) == 0
        assert len(pool) == 0

    def test_from_wiki_handles_none_structured_and_evidence(self, pool):
        logic = MagicMock()
        logic.name = "a"
        logic.extracted_formula = "f"
        logic.source_detail = None  # 触发 defaults
        logic.created_at = None
        logic.wiki_page_name = None
        logic.structured = None
        logic.performance_evidence = None

        proxy = MagicMock()
        proxy.list_logics.return_value = [logic]
        pool.from_wiki(proxy)
        e = pool.get("a")
        assert e is not None
        assert e.source_lib == "wiki"
        assert e.ir == 0.0

    def test_to_wiki_writes_each_entry(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006", ir=0.7))
        pool.add(make_entry("alpha191-alpha001", ir=0.5))

        proxy = MagicMock()
        proxy.store_logic.return_value = "Logic/xxx.md"
        n = pool.to_wiki(proxy)
        assert n == 2
        assert proxy.store_logic.call_count == 2
        assert pool.get("alpha101-alpha006").wiki_path == "Logic/xxx.md"

    def test_to_wiki_continues_on_failure(self, pool, make_entry):
        pool.add(make_entry("alpha101-alpha006", ir=0.7))
        pool.add(make_entry("alpha191-alpha001", ir=0.5))

        proxy = MagicMock()
        proxy.store_logic.side_effect = [
            "Logic/a.md",
            RuntimeError("disk error"),
        ]
        n = pool.to_wiki(proxy)
        assert n == 1
        failures = pool.failed_writes()
        assert len(failures) == 1
        assert failures[0]["formula_id"] == "alpha191-alpha001"
        assert "disk error" in failures[0]["error"]

    def test_to_wiki_skips_when_pool_empty(self, pool):
        proxy = MagicMock()
        n = pool.to_wiki(proxy)
        assert n == 0
        assert proxy.store_logic.call_count == 0


# ======================================================================
# JSON 离线持久化
# ======================================================================
class TestJsonPersistence:
    def test_save_load_roundtrip(self, pool, make_entry, tmp_path: Path):
        pool.add(make_entry("alpha101-alpha006", ir=0.7, structured={"predicates": []}))
        pool.add(make_entry("alpha191-alpha001", ir=0.5, tags=["logic-driven"]))

        path = tmp_path / "pool.json"
        pool.save_json(path)

        loaded = FactorPool.load_json(path)
        assert len(loaded) == 2
        assert loaded.wiki_path == pool.wiki_path
        assert loaded.get("alpha101-alpha006").ir == 0.7
        assert loaded.get("alpha191-alpha001").tags == ["logic-driven"]

    def test_save_creates_parent_dirs(self, pool, make_entry, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "pool.json"
        pool.add(make_entry("alpha101-alpha006"))
        pool.save_json(path)
        assert path.exists()

    def test_load_with_failed_writes(self, pool, make_entry, tmp_path: Path):
        pool.add(make_entry("alpha101-alpha006"))
        proxy = MagicMock()
        proxy.store_logic.side_effect = RuntimeError("x")
        pool.to_wiki(proxy)

        path = tmp_path / "pool.json"
        pool.save_json(path)
        loaded = FactorPool.load_json(path)
        assert len(loaded.failed_writes()) == 1

    def test_load_json_preserves_structure(self, pool, make_entry, tmp_path: Path):
        structured = {
            "predicates": [{"variable": "close", "op": "rank", "threshold": 0.5, "weight": 1.0}],
            "behavior": {"target": "forward_return_5", "direction": -1, "horizon": 5},
        }
        pool.add(make_entry("alpha101-alpha006", structured=structured))
        path = tmp_path / "pool.json"
        pool.save_json(path)
        loaded = FactorPool.load_json(path)
        assert loaded.get("alpha101-alpha006").structured == structured


# ======================================================================
# 线程安全
# ======================================================================
class TestThreadSafety:
    def test_concurrent_add_is_safe(self, make_entry):
        pool = FactorPool(wiki_path="/tmp/wiki_test")
        n_threads = 8
        per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(per_thread):
                pool.add(make_entry(f"alpha101-t{tid}-i{i}", ir=tid * 0.01))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(pool) == n_threads * per_thread
        # 验证 entries 都还在 (无丢失)
        assert pool.get("alpha101-t0-i0") is not None
        assert pool.get("alpha101-t7-i49") is not None

    def test_concurrent_read_write(self, make_entry):
        pool = FactorPool(wiki_path="/tmp/wiki_test")
        for i in range(20):
            pool.add(make_entry(f"alpha101-prefill{i}", ir=0.5))

        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                pool.add(make_entry(f"alpha101-w{i}", ir=0.5))
                i += 1

        def reader():
            while not stop.is_set():
                _ = pool.summary()
                _ = pool.select(top_n=5)

        t_w = threading.Thread(target=writer)
        t_r1 = threading.Thread(target=reader)
        t_r2 = threading.Thread(target=reader)
        t_w.start()
        t_r1.start()
        t_r2.start()
        import time

        time.sleep(0.2)
        stop.set()
        t_w.join()
        t_r1.join()
        t_r2.join()
        # No exceptions raised = success