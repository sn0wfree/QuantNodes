"""Knowledge RAG 模块测试 (15 tests)。

覆盖:
    - IdentityRetriever (3)
    - TFIDFRetriever (2)
    - KnowledgeBase (5)
    - build_rag_prompt (3)
    - EvolutionLoop 集成 (2)
"""
from __future__ import annotations


import pytest

from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionSetting,
    FactorCandidate,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    IdentityRetriever,
    KnowledgeBase,
    TFIDFRetriever,
    build_rag_prompt,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# Fixtures
# ============================================================================

def _make_entry(
    eid: str,
    name: str = "f",
    expression: str = "close - open",
    hypothesis: str = "h",
    description: str = "d",
    round_idx: int = 0,
    operation: str = "original",
    parent_ids: list | None = None,
    sharpe: float = 1.0,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid,
        round_idx=round_idx,
        operation=operation,
        parent_ids=parent_ids or [],
        config_snapshot={
            "factor": {
                "name": name,
                "expression": expression,
                "hypothesis": hypothesis,
                "description": description,
            },
        },
        feedback=FactorFeedback(factor_name=name, decision=True, summary="ok"),
        metrics={"sharpe": sharpe, "arr": sharpe * 0.1, "ic_mean": 0.04},
    )


@pytest.fixture
def small_pool() -> list[TrajectoryEntry]:
    return [
        _make_entry("e1", name="momentum_20d", hypothesis="momentum",
                    expression="(close-close.shift(20))/close.shift(20)",
                    description="20-day price momentum", sharpe=1.5),
        _make_entry("e2", name="reversal_5d", hypothesis="reversal",
                    expression="close - close.shift(5)",
                    description="5-day reversal", sharpe=1.0),
        _make_entry("e3", name="momentum_60d", hypothesis="momentum",
                    expression="(close-close.shift(60))/close.shift(60)",
                    description="60-day momentum variant", parent_ids=["e1"],
                    round_idx=1, operation="mutation", sharpe=1.2),
    ]


# ============================================================================
# 1. IdentityRetriever (3)
# ============================================================================

def test_identity_retriever_basic():
    """IdentityRetriever 基础添加 + 查询。"""
    r = IdentityRetriever()
    r.add("a", "momentum factor using close prices")
    r.add("b", "volatility factor using returns")
    r.add("c", "reversal factor using close prices")
    results = r.query("momentum", top_k=3)
    assert len(results) >= 1
    assert results[0][0] == "a"


def test_identity_retriever_empty():
    """空 retriever 查询返回空。"""
    r = IdentityRetriever()
    assert r.query("anything") == []
    assert len(r) == 0


def test_identity_retriever_update_existing():
    """更新已存在的 doc_id (覆盖文本)。"""
    r = IdentityRetriever()
    r.add("a", "momentum factor")
    r.add("a", "reversal factor")  # 覆盖
    assert len(r) == 1
    results = r.query("momentum", top_k=1)
    assert results == []  # momentum 已无对应文本


# ============================================================================
# 2. TFIDFRetriever (2)
# ============================================================================

def test_tfidf_retriever_basic():
    """TFIDFRetriever 基础添加 + 查询 (sklearn)。"""
    r = TFIDFRetriever()
    r.add("a", "momentum factor using close prices")
    r.add("b", "volatility factor using returns")
    results = r.query("momentum close", top_k=2)
    assert results[0][0] == "a"
    assert results[0][1] > 0


def test_tfidf_retriever_empty():
    """空 TFIDFRetriever。"""
    r = TFIDFRetriever()
    assert r.query("x") == []


# ============================================================================
# 3. KnowledgeBase (5)
# ============================================================================

def test_kb_add_and_query(tmp_path, small_pool):
    """KB 添加 + 查询。"""
    pool = TrajectoryPool(tmp_path)
    for e in small_pool:
        pool.add(e)
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    for e in small_pool:
        kb.add(e)
    assert len(kb) == 3
    results = kb.query("momentum", top_k=2)
    assert len(results) == 2
    # momentum 类的 entry 应排前
    assert "momentum" in results[0][0].config_snapshot["factor"]["hypothesis"]


def test_kb_query_with_lineage(tmp_path, small_pool):
    """query_with_lineage 包含 parents/children 上下文。"""
    pool = TrajectoryPool(tmp_path)
    for e in small_pool:
        pool.add(e)
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    kb.sync_from_pool()

    # e3 是 e1 的 mutation, 检索 e3 的 query "momentum"
    results = kb.query_with_lineage("momentum", top_k=2)
    e3 = next(r for r in results if r["entry"].entry_id == "e3")
    assert len(e3["parents"]) == 1
    assert e3["parents"][0].entry_id == "e1"
    assert len(e3["children"]) == 0


def test_kb_sync_from_pool(tmp_path, small_pool):
    """sync_from_pool 增量同步。"""
    pool = TrajectoryPool(tmp_path)
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    n1 = kb.sync_from_pool()
    assert n1 == 0  # pool 空

    for e in small_pool:
        pool.add(e)
    n2 = kb.sync_from_pool()
    assert n2 == 3
    assert len(kb) == 3

    # 再次同步, 0 新增
    n3 = kb.sync_from_pool()
    assert n3 == 0


def test_kb_save_load(tmp_path, small_pool):
    """KB 索引可持久化。"""
    pool = TrajectoryPool(tmp_path / "pool")
    for e in small_pool:
        pool.add(e)
    kb1 = KnowledgeBase(IdentityRetriever(), pool=pool)
    kb1.sync_from_pool()

    path = tmp_path / "kb.parquet"
    kb1.save(path)
    assert path.exists()

    kb2 = KnowledgeBase.load(path, pool=pool)
    assert len(kb2) == 3
    results = kb2.query("momentum", top_k=2)
    assert len(results) >= 1


def test_kb_min_score_filter(small_pool):
    """min_score 过滤低分结果。"""
    kb = KnowledgeBase(IdentityRetriever())
    for e in small_pool:
        kb.add(e)
    # 不相关 query, 应被过滤
    results = kb.query("xyz_qq_zzz_unrelated", top_k=5, min_score=0.5)
    assert len(results) == 0


# ============================================================================
# 4. build_rag_prompt (3)
# ============================================================================

def test_rag_prompt_with_kb(tmp_path, small_pool):
    """有 KB 时, prompt 含示例段。"""
    pool = TrajectoryPool(tmp_path)
    for e in small_pool:
        pool.add(e)
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    for e in small_pool:
        kb.add(e)
    prompt = build_rag_prompt("momentum", "20-day momentum", kb=kb, top_k=2)
    assert "示例 1" in prompt
    assert "momentum" in prompt.lower()
    assert "momentum_20d" in prompt  # 检索到的示例


def test_rag_prompt_without_kb():
    """无 KB 时, prompt 简洁不含示例段。"""
    prompt = build_rag_prompt("momentum", "20-day momentum", kb=None, top_k=3)
    assert "示例 1" not in prompt
    assert "momentum" in prompt  # 仍含任务段


def test_rag_prompt_empty_kb():
    """空 KB 时退化为无示例模式。"""
    kb = KnowledgeBase(IdentityRetriever())  # 空
    prompt = build_rag_prompt("momentum", "20-day momentum", kb=kb, top_k=3)
    assert "示例 1" not in prompt


# ============================================================================
# 5. EvolutionLoop 集成 (2)
# ============================================================================

def test_evolution_loop_with_kb(tmp_path, small_pool):
    """EvolutionLoop 接受 knowledge_base, round 1 检索能用到 round 0。"""
    pool = TrajectoryPool(tmp_path / "pool")
    # 预填 1 个 momentum entry (模拟历史)
    pool.add(small_pool[0])
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    kb.sync_from_pool()

    def evaluate(c: FactorCandidate) -> tuple[bool, dict, FactorFeedback]:
        return (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name,
            decision=True, summary="ok",
        ))

    settings = EvolutionSetting(enabled=True, max_rounds=1, seed=42)
    loop = EvolutionLoop(
        settings, pool=pool, evaluate_fn=evaluate,
        knowledge_base=kb, rag_top_k=2,
    )
    # Round 0 用 KB
    captured_prompts: list[str] = []

    class CapturingHypothesizer:
        def __init__(self, real):
            self._real = real

        def hypothesize(self, direction, description=""):
            # 捕获 prompt
            from QuantNodes.core.knowledge import build_rag_prompt
            captured_prompts.append(build_rag_prompt(direction, description, self._real.knowledge_base, self._real.rag_top_k))
            return self._real.hypothesize(direction, description)

    loop.hypothesizer = CapturingHypothesizer(loop.hypothesizer)
    loop.run(initial_directions=["momentum"])
    assert any("示例 1" in p for p in captured_prompts)


def test_evolution_loop_sync_kb(tmp_path, small_pool):
    """sync_knowledge_base() 手动同步接口可用。"""
    pool = TrajectoryPool(tmp_path)
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    settings = EvolutionSetting(enabled=True, max_rounds=0)
    loop = EvolutionLoop(
        settings, pool=pool,
        evaluate_fn=lambda c: (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name, decision=True, summary="ok",
        )),
        knowledge_base=kb,
    )
    for e in small_pool:
        pool.add(e)
    n = loop.sync_knowledge_base()
    assert n == 3
    assert len(loop.knowledge_base) == 3


# ============================================================================
# H19: KnowledgeBaseSetting 字段权重可配置
# ============================================================================

from QuantNodes.core.knowledge.knowledge_base import (
    DEFAULT_FIELD_WEIGHTS,
    KnowledgeBaseSetting,
)


class TestKnowledgeBaseSetting:
    def test_default_field_weights(self):
        s = KnowledgeBaseSetting()
        assert s.field_weights == DEFAULT_FIELD_WEIGHTS
        assert s.field_weights["name"] == 3.0
        assert s.field_weights["hypothesis"] == 2.5

    @pytest.mark.parametrize("override", [
        {"name": 10.0},  # 提权 name
        {"name": 1.0, "expression": 1.0},  # 全部降权
        {},  # 空
    ])
    def test_custom_field_weights(self, override):
        s = KnowledgeBaseSetting(field_weights=override)
        assert s.field_weights == override

    def test_setting_constructs_kb(self):
        from QuantNodes.core.knowledge import KnowledgeBase
        s = KnowledgeBaseSetting(field_weights={"name": 10.0})
        kb = KnowledgeBase(setting=s)
        assert kb._field_weights["name"] == 10.0

    def test_kwarg_field_weights_compat(self):
        """旧 API field_weights= 仍可用。"""
        from QuantNodes.core.knowledge import KnowledgeBase
        kb = KnowledgeBase(field_weights={"name": 5.0})
        assert kb._field_weights["name"] == 5.0

    def test_setting_overrides_kwarg(self):
        """setting 优先于 field_weights。"""
        from QuantNodes.core.knowledge import KnowledgeBase
        s = KnowledgeBaseSetting(field_weights={"name": 10.0})
        kb = KnowledgeBase(setting=s, field_weights={"name": 1.0})
        assert kb._field_weights["name"] == 10.0  # setting 胜

    def test_text_construction_uses_custom_weights(self, tmp_path):
        """自定义权重影响 _entry_to_text 输出。"""
        from QuantNodes.core.knowledge import KnowledgeBase
        from QuantNodes.core.feedback import FactorFeedback
        from QuantNodes.core.trajectory import TrajectoryEntry

        kb = KnowledgeBase(field_weights={"name": 5.0, "summary": 0})
        entry = TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"factor": {"name": "test_name", "expression": "close"}},
            feedback=FactorFeedback(
                factor_id="e1", factor_name="test_name",
                decision=True, summary="ok",
            ),
        )
        text = kb._entry_to_text(entry)
        # name 重复 5 次
        assert text.count("test_name") == 5
        # summary 重复 0 次 (默认 1.0)
        assert "ok" not in text


class TestDefaultFieldWeights:
    def test_default_has_5_keys(self):
        assert len(DEFAULT_FIELD_WEIGHTS) == 5

    @pytest.mark.parametrize("key", ["name", "expression", "hypothesis", "description", "summary"])
    def test_required_keys(self, key):
        assert key in DEFAULT_FIELD_WEIGHTS

    @pytest.mark.parametrize("w", [0.0, 1.0, 2.5, 3.0, 10.0])
    def test_custom_weight_value(self, w):
        from QuantNodes.core.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(field_weights={"name": w})
        assert kb._field_weights["name"] == w
