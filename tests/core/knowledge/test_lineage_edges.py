"""lineage_compress.py / lineage_expand.py 边界测试 (15 tests)。

聚焦:
    - Compressor.heuristic: 空、单元素、多元素、relation=ancestors vs descendants
    - Compressor.heuristic: max_tokens 截断、缺 config_snapshot、缺 metrics
    - Compressor.LLM: 正常 JSON、JSON 错误 fallback、无 callable 真实 model
    - expand_lineage: 不存在 root、空 pool、多层谱系、max_ancestors 限制
    - expand_lineage_batch: 去重
    - compress_lineage 便捷函数
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import Compressor, compress_lineage
from QuantNodes.core.knowledge.lineage_expand import (
    expand_lineage,
    expand_lineage_batch,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


def _entry(
    entry_id: str, round_idx: int = 0, op: str = "original",
    parent_ids: list[str] | None = None, sharpe: float = 0.5,
    name: str = "f", config: dict | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id, round_idx=round_idx,
        operation=op, parent_ids=parent_ids or [],
        config_snapshot=config or {"factor": {"name": name, "expression": "close"}},
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=name,
            decision=True, summary="ok",
        ),
        metrics={"sharpe": sharpe},
    )


# ============================================================================
# 1. Compressor.heuristic (6 tests)
# ============================================================================

class TestCompressorHeuristic:
    def test_empty_entries(self):
        c = Compressor(model="mock")
        r = c.compress([], relation="ancestors")
        assert r.summary == ""
        assert r.original_count == 0
        assert r.compressed_chars == 0
        assert r.method == "heuristic"

    def test_single_entry(self):
        c = Compressor(model="mock")
        e = _entry("e1", name="alpha")
        r = c.compress([(1, e)], relation="ancestors")
        assert r.original_count == 1
        assert "alpha" in r.summary
        assert "↑" in r.summary  # ancestors 箭头
        assert r.method == "heuristic"

    def test_descendants_relation(self):
        c = Compressor(model="mock")
        e = _entry("e1", name="beta")
        r = c.compress([(1, e)], relation="descendants")
        assert "↓" in r.summary
        assert "beta" in r.summary

    def test_accepts_plain_entries_not_tuples(self):
        """compress 也接受非 tuple 列表 (默认 depth=0)。"""
        c = Compressor(model="mock")
        e = _entry("e1", name="gamma")
        r = c.compress([e], relation="ancestors")
        assert r.original_count == 1
        assert "gamma" in r.summary

    def test_max_tokens_truncates(self):
        c = Compressor(model="mock", max_tokens=20)
        entries = [(_entry(f"e{i}", name=f"name_{i}"),) for i in range(10)]
        # 转成 (depth, entry) 格式
        normalized = [(0, e[0]) for e in entries]
        r = c.compress(normalized, relation="ancestors")
        assert len(r.summary) <= 20
        assert r.summary.endswith("...")  # 截断标识

    def test_missing_config_snapshot(self):
        """config_snapshot 为空 → 用 entry_id[:8] 兜底。"""
        c = Compressor(model="mock")
        e = TrajectoryEntry(
            entry_id="abcdef12345",
            config_snapshot={},  # 空
            metrics={},  # 也空
            feedback=None,
        )
        r = c.compress([(1, e)], relation="ancestors")
        # 用 entry_id 前 8 字符
        assert "abcdef12" in r.summary
        # sharpe 默认 0
        assert "sharpe=0.00" in r.summary

    def test_missing_metrics(self):
        c = Compressor(model="mock")
        e = TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"factor": {"name": "x"}},
            metrics=None,
            feedback=None,
        )
        r = c.compress([(1, e)], relation="ancestors")
        assert "sharpe=0.00" in r.summary


# ============================================================================
# 2. Compressor.LLM 模式 (4 tests)
# ============================================================================

class TestCompressorLLM:
    def test_llm_callable_returns_json(self):
        def fake_llm(prompt):
            return json.dumps({"summary": "这是 LLM 总结"})
        c = Compressor(model="mock", llm_callable=fake_llm)
        r = c.compress([(1, _entry("e1", name="x"))], relation="ancestors")
        assert r.summary == "这是 LLM 总结"
        assert r.method == "llm"

    def test_llm_callable_invalid_json_fallback(self):
        def bad_llm(prompt):
            return "not json"
        c = Compressor(model="mock", llm_callable=bad_llm)
        r = c.compress([(1, _entry("e1", name="alpha"))], relation="ancestors")
        # fallback to heuristic
        assert r.method == "heuristic"
        assert "alpha" in r.summary

    def test_real_model_auto_injects_gateway(self):
        """model='deepseek' 无 llm_callable → 自动注入 LLMGateway。"""
        from QuantNodes.ai.llm.gateway import LLMGateway
        c = Compressor(model="deepseek-v3")
        assert isinstance(c._llm_callable, LLMGateway)

    def test_llm_prompt_includes_entries(self):
        captured = {}

        def fake_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({"summary": "ok"})

        c = Compressor(model="mock", llm_callable=fake_llm)
        c.compress([
            (1, _entry("e1", name="alpha_one", sharpe=0.5)),
            (2, _entry("e2", name="alpha_two", sharpe=1.0)),
        ], relation="ancestors")
        # prompt 包含 entry 详情
        assert "alpha_one" in captured["prompt"]
        assert "alpha_two" in captured["prompt"]
        assert "ancestors" in captured["prompt"]


# ============================================================================
# 3. compress_lineage 便捷函数 (2 tests)
# ============================================================================

class TestCompressLineageFn:
    def test_module_level_function(self):
        r = compress_lineage([(1, _entry("e1", name="x"))])
        assert r.summary
        assert r.method == "heuristic"

    def test_with_custom_model(self):
        def fake_llm(prompt):
            return json.dumps({"summary": "custom summary"})
        r = compress_lineage(
            [(1, _entry("e1"))], model="custom", llm_callable=fake_llm,
        )
        assert r.summary == "custom summary"
        assert r.method == "llm"


# ============================================================================
# 4. expand_lineage (5 tests)
# ============================================================================

class TestExpandLineage:
    def _make_pool(self, tmp_path: Path) -> TrajectoryPool:
        pool = TrajectoryPool(tmp_path)
        # 3 层谱系: gp → p → c → gc
        pool.add(_entry("gp", round_idx=0, op="original"))
        pool.add(_entry("p", round_idx=1, op="mutation", parent_ids=["gp"]))
        pool.add(_entry("c", round_idx=2, op="mutation", parent_ids=["p"]))
        pool.add(_entry("gc", round_idx=3, op="crossover", parent_ids=["c"]))
        return pool

    def test_missing_root_id(self, tmp_path: Path):
        pool = self._make_pool(tmp_path)
        r = expand_lineage(pool, "missing")
        assert r == {"root": None, "ancestors": [], "descendants": []}

    def test_root_with_no_relatives(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("lonely"))
        r = expand_lineage(pool, "lonely")
        assert r["root"].entry_id == "lonely"
        assert r["ancestors"] == []
        assert r["descendants"] == []

    def test_ancestors_depth(self, tmp_path: Path):
        pool = self._make_pool(tmp_path)
        r = expand_lineage(pool, "gc", max_ancestor_depth=3, max_descendant_depth=0)
        ancestors = r["ancestors"]
        # depth=1: c, depth=2: p, depth=3: gp
        depths = {d: a.entry_id for d, a in ancestors}
        assert depths == {1: "c", 2: "p", 3: "gp"}

    def test_ancestor_depth_limit(self, tmp_path: Path):
        pool = self._make_pool(tmp_path)
        r = expand_lineage(pool, "gc", max_ancestor_depth=1)
        # 只到 p 不再到 gp
        ancestors = [a.entry_id for _, a in r["ancestors"]]
        assert "c" in ancestors
        assert "p" not in ancestors
        assert "gp" not in ancestors

    def test_descendants_depth(self, tmp_path: Path):
        pool = self._make_pool(tmp_path)
        r = expand_lineage(pool, "gp", max_ancestor_depth=0, max_descendant_depth=3)
        descendants = r["descendants"]
        depths = {dep: d.entry_id for dep, d in descendants}
        assert depths == {1: "p", 2: "c", 3: "gc"}

    def test_max_ancestors_limit(self, tmp_path: Path):
        """max_ancestors 限制数量, 防止 token 爆炸。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("root"))
        # 10 个 parent 链
        prev = "root"
        for i in range(10):
            cur = f"p{i}"
            pool.add(_entry(cur, round_idx=i+1, parent_ids=[prev]))
            prev = cur
        r = expand_lineage(pool, prev, max_ancestor_depth=10, max_ancestors=3)
        # 最多 3 个 ancestor
        assert len(r["ancestors"]) <= 3


# ============================================================================
# 5. expand_lineage_batch (2 tests)
# ============================================================================

class TestExpandLineageBatch:
    def test_batch_dedup(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("a"))
        pool.add(_entry("b"))
        results = expand_lineage_batch(pool, ["a", "a", "b", "a"])
        # 去重后 2 个
        assert len(results) == 2

    def test_batch_with_missing(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("a"))
        results = expand_lineage_batch(pool, ["a", "missing"])
        # missing 返回空 dict, a 正常
        assert len(results) == 2
        roots = [r["root"] for r in results]
        assert None in roots  # missing
        assert any(r.entry_id == "a" for r in roots if r is not None)
