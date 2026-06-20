"""knowledge 模块边界条件测试 (20 tests)。

聚焦:
    - TFIDFRetriever: 空 query、单 doc、重复 doc_id (覆盖)、k>n
    - IdentityRetriever: 纯 Python TF-IDF, 空/单/多
    - make_retriever: 未知 kind 抛 ValueError
    - KnowledgeBase: 增/查/sync_from_pool/save/load
    - 字段权重: name 重复 3 次 → score 应较高
    - 空 KB 不崩
"""
from __future__ import annotations

from pathlib import Path

import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    KnowledgeBase,
    make_retriever,
)
from QuantNodes.core.knowledge.retriever import (
    IdentityRetriever,
    TFIDFRetriever,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# 1. TFIDFRetriever (4 tests)
# ============================================================================

class TestTFIDFRetriever:
    def test_empty_returns_empty(self):
        r = TFIDFRetriever()
        assert r.query("anything") == []
        assert len(r) == 0

    def test_single_doc(self):
        r = TFIDFRetriever()
        r.add("d1", "momentum reversal alpha")
        result = r.query("momentum", top_k=5)
        assert len(result) == 1
        assert result[0][0] == "d1"
        assert result[0][1] > 0.0

    def test_duplicate_doc_id_overwrites(self):
        """add 重复 id → 覆盖 (而非重复)。"""
        r = TFIDFRetriever()
        r.add("d1", "momentum")
        r.add("d1", "volume")  # 覆盖
        assert len(r) == 1
        # 查询 "momentum" 应无结果
        result = r.query("momentum")
        assert result == []
        # 查询 "volume" 应有
        result = r.query("volume")
        assert result == [("d1", result[0][1])]

    def test_top_k_limits_results(self):
        r = TFIDFRetriever()
        for i in range(10):
            r.add(f"d{i}", f"alpha beta gamma {i}")
        result = r.query("alpha", top_k=3)
        assert len(result) == 3

    def test_query_filtered_by_score(self):
        """score=0 的文档被过滤。"""
        r = TFIDFRetriever()
        r.add("d1", "momentum")
        r.add("d2", "volume")  # 无 momentum 关键词
        result = r.query("momentum")
        # d2 应被过滤 (score=0)
        ids = [d for d, _ in result]
        assert "d1" in ids
        assert "d2" not in ids


# ============================================================================
# 2. IdentityRetriever (3 tests)
# ============================================================================

class TestIdentityRetriever:
    def test_empty(self):
        r = IdentityRetriever()
        assert r.query("anything") == []

    def test_basic_match(self):
        r = IdentityRetriever()
        r.add("d1", "momentum reversal")
        r.add("d2", "volume volatility")
        result = r.query("momentum")
        assert result[0][0] == "d1"
        assert result[0][1] > 0.0

    def test_relevance_scoring(self):
        """关键词匹配多的文档得分高。"""
        r = IdentityRetriever()
        r.add("d1", "momentum alpha momentum momentum")
        r.add("d2", "momentum volume")
        result = r.query("momentum")
        # d1 含 momentum 更多
        assert result[0][0] == "d1"


# ============================================================================
# 3. make_retriever (2 tests)
# ============================================================================

class TestMakeRetriever:
    def test_tfidf_kind(self):
        r = make_retriever("tfidf")
        assert isinstance(r, TFIDFRetriever)

    def test_identity_kind(self):
        r = make_retriever("identity")
        assert isinstance(r, IdentityRetriever)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未知 retriever kind"):
            make_retriever("bogus")


# ============================================================================
# 4. KnowledgeBase (8 tests)
# ============================================================================

def _make_entry(
    entry_id: str, name: str, expression: str = "close",
    hypothesis: str = "", description: str = "",
    sharpe: float = 0.5, round_idx: int = 0,
    parent_ids: list[str] | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id,
        round_idx=round_idx,
        parent_ids=parent_ids or [],
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=name,
            decision=True, summary=f"sharpe={sharpe}",
        ),
        config_snapshot={"factor": {
            "name": name, "expression": expression,
            "hypothesis": hypothesis, "description": description,
        }},
        metrics={"sharpe": sharpe},
    )


class TestKnowledgeBase:
    def test_empty_kb_query(self):
        kb = KnowledgeBase()
        assert len(kb) == 0
        result = kb.query("anything")
        assert result == []

    def test_add_and_query(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "mom_20", "close - close.shift(20)",
                             description="momentum reversal factor"))
        pool.add(_make_entry("e2", "vol_osc", "volume.diff(5)",
                             description="volume oscillator"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        assert len(kb) == 2
        # 用 description 关键词, 因 TFIDF \w+ 不拆 _
        result = kb.query("momentum", top_k=1)
        assert result[0][0].entry_id == "e1"

    def test_name_weight_3x(self, tmp_path: Path):
        """name 字段权重 3 → 提升检索相关度。"""
        pool = TrajectoryPool(tmp_path)
        # e1: "alpha" 出现 1 次 (在 name)
        pool.add(_make_entry("e1", "alpha", "close"))
        # e2: "alpha" 出现 1 次 (在 description)
        pool.add(_make_entry("e2", "beta", "close", description="alpha alpha"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        result = kb.query("alpha", top_k=1)
        # e1.name="alpha" 重复 3 次 (权重 3) → 应胜出
        assert result[0][0].entry_id == "e1"

    def test_min_score_filter(self, tmp_path: Path):
        """min_score 过滤低分结果。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "momentum_20", "close"))
        pool.add(_make_entry("e2", "volume_osc", "close"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        # 设极高 min_score
        result = kb.query("momentum", top_k=5, min_score=0.99)
        # 全被过滤
        assert result == []

    def test_sync_from_pool(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "alpha", "close"))
        kb = KnowledgeBase(pool=pool)
        assert len(kb) == 0
        n = kb.sync_from_pool()
        assert n == 1
        assert len(kb) == 1

    def test_sync_idempotent(self, tmp_path: Path):
        """重复 sync 不增加。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "alpha", "close"))
        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()
        n = kb.sync_from_pool()
        assert n == 0  # 已索引
        assert len(kb) == 1

    def test_add_many_returns_new_count(self, tmp_path: Path):
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "a", "close"))
        pool.add(_make_entry("e2", "b", "close"))
        kb = KnowledgeBase(pool=pool)
        n = kb.add_many([pool.get("e1"), pool.get("e2")])
        assert n == 2
        # 再次 add 同样 → 0
        n = kb.add_many([pool.get("e1")])
        assert n == 0

    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "kb.parquet"
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("e1", "alpha", "close"))
        kb1 = KnowledgeBase(pool=pool)
        kb1.add(pool.get("e1"))
        kb1.save(path)
        # 加载
        kb2 = KnowledgeBase.load(path, pool=pool)
        assert len(kb2) == 1
        result = kb2.query("alpha", top_k=1)
        assert result[0][0].entry_id == "e1"

    def test_query_with_lineage(self, tmp_path: Path):
        """谱系查询: parents/children/ancestors/descendants。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_make_entry("p", "parent", "close", round_idx=0))
        pool.add(_make_entry("c", "child", "close.diff(5)",
                             hypothesis="momentum", round_idx=1, parent_ids=["p"]))
        pool.add(_make_entry("gc", "grandchild", "close.diff(10)",
                              round_idx=2, parent_ids=["c"]))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        results = kb.query_with_lineage("momentum", top_k=3, max_ancestor_depth=2)
        # 至少一个 entry 是 c
        found_c = False
        for r in results:
            if r["entry"] and r["entry"].entry_id == "c":
                found_c = True
                # parents (depth=1) 应包含 p
                parent_ids = [p.entry_id for p in r["parents"]]
                assert "p" in parent_ids
        assert found_c
