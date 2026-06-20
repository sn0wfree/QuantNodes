"""EvolutionLoop 模块测试 (20 tests)。

覆盖:
    - FactorCandidate + Settings (3)
    - Hypothesizer (3)
    - Mutator (3)
    - Crosser (2)
    - EvolutionLoop 主循环 (6)
    - 与 QualityGate 集成 (2)
    - 与 TrajectoryPool 集成 (1)
"""
from __future__ import annotations

import json
from typing import Callable

import pytest

from QuantNodes.core.evolution import (
    Crosser,
    EvolutionLoop,
    EvolutionSetting,
    FactorCandidate,
    Hypothesizer,
    Mutator,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.quality_gate import (
    FactorZoo,
    QualityGateNode,
    QualityGateSetting,
    RedundancySetting,
)
from QuantNodes.core.trajectory import (
    TrajectoryPool,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_evaluate(metric_value: float = 0.5, passed: bool = True) -> Callable:
    def evaluate(candidate: FactorCandidate) -> tuple[bool, dict, FactorFeedback]:
        return (
            passed,
            {"sharpe": metric_value, "arr": metric_value / 2, "ic_mean": 0.04},
            FactorFeedback(
                factor_id=candidate.factor_id,
                factor_name=candidate.name,
                decision=passed,
                summary="mock",
            ),
        )
    return evaluate


# ============================================================================
# 1. FactorCandidate + Settings (3)
# ============================================================================

def test_factor_candidate_creation():
    """FactorCandidate 基础。"""
    c = FactorCandidate(
        factor_id="abc",
        name="momentum",
        expression="close - open",
        hypothesis="h",
        description="d",
    )
    assert c.factor_id == "abc"
    assert c.expression == "close - open"
    assert c.hypothesis == "h"


def test_evolution_setting_defaults():
    """默认配置合理。"""
    s = EvolutionSetting()
    assert s.enabled is False
    assert s.max_rounds == 3
    assert s.parent_selection_strategy == "top_percent_plus_random"


def test_evolution_setting_any_operator_enabled():
    """any_operator_enabled() 判断。"""
    s = EvolutionSetting()
    # 3 个 operator 默认 enabled=True
    assert s.any_operator_enabled() is True
    s.hypothesizer.enabled = False
    s.mutator.enabled = False
    s.crosser.enabled = False
    assert s.any_operator_enabled() is False


# ============================================================================
# 2. Hypothesizer (3)
# ============================================================================

def test_hypothesizer_mock():
    """mock 模式生成候选。"""
    h = Hypothesizer(model="mock", seed=42)
    c = h.hypothesize(direction="momentum", description="20-day")
    assert isinstance(c, FactorCandidate)
    assert c.hypothesis == "momentum"
    assert c.expression
    assert c.factor_id


def test_hypothesizer_custom_callable():
    """支持自定义 llm_callable。"""
    def fake_llm(prompt):
        return json.dumps({
            "name": "custom",
            "expression": "close - open",
            "description": "custom desc",
        })
    h = Hypothesizer(llm_callable=fake_llm)
    c = h.hypothesize(direction="h", description="d")
    assert c.name == "custom"
    assert c.expression == "close - open"


def test_hypothesizer_parse_fallback():
    """解析失败时 mock 兜底。"""
    def bad_llm(prompt):
        return "not json"
    h = Hypothesizer(llm_callable=bad_llm, max_correction_attempts=2)
    c = h.hypothesize(direction="h", description="d")
    # 兜底后仍能生成候选
    assert c.expression


# ============================================================================
# 3. Mutator (3)
# ============================================================================

def test_mutator_mock():
    """mock mutate 产生变体。"""
    m = Mutator(model="mock", seed=42)
    parent = FactorCandidate(factor_id="p", name="p", expression="close - open")
    child = m.mutate(parent)
    assert child.factor_id != parent.factor_id
    assert child.expression
    assert child.hypothesis == parent.hypothesis


def test_mutator_preserves_hypothesis():
    """mutation 保留 hypothesis。"""
    m = Mutator(model="mock", seed=1)
    parent = FactorCandidate(
        factor_id="p", name="p", expression="close - open",
        hypothesis="momentum effect", description="20-day",
    )
    child = m.mutate(parent)
    assert child.hypothesis == "momentum effect"
    # description 由 mock 生成 (真实 LLM 也会生成新 description)
    assert child.description != ""


def test_mutator_custom_callable():
    """支持自定义 llm_callable。"""
    def fake_llm(prompt):
        return json.dumps({
            "name": "mutated",
            "expression": "open - close",
            "description": "mut desc",
        })
    m = Mutator(llm_callable=fake_llm)
    parent = FactorCandidate(factor_id="p", name="p", expression="close - open")
    child = m.mutate(parent)
    assert child.name == "mutated"
    assert child.expression == "open - close"


# ============================================================================
# 4. Crosser (2)
# ============================================================================

def test_crosser_mock():
    """mock crossover 产生组合。"""
    cr = Crosser(model="mock", seed=42)
    p1 = FactorCandidate(factor_id="p1", name="p1", expression="close - open")
    p2 = FactorCandidate(factor_id="p2", name="p2", expression="volume / 1000")
    child = cr.crossover(p1, p2)
    assert child.expression
    assert "combo" in child.hypothesis


def test_crosser_custom_callable():
    """支持自定义 llm_callable。"""
    def fake_llm(prompt):
        return json.dumps({
            "name": "combo",
            "expression": "p1 + p2",
            "description": "sum",
        })
    cr = Crosser(llm_callable=fake_llm)
    p1 = FactorCandidate(factor_id="p1", name="p1", expression="a")
    p2 = FactorCandidate(factor_id="p2", name="p2", expression="b")
    child = cr.crossover(p1, p2)
    assert child.name == "combo"
    assert child.expression == "p1 + p2"


# ============================================================================
# 5. EvolutionLoop 主循环 (6)
# ============================================================================

def test_loop_requires_evaluate_fn(tmp_path):
    """无 evaluate_fn 抛错。"""
    pool = TrajectoryPool(tmp_path)
    with pytest.raises(ValueError, match="evaluate_fn"):
        EvolutionLoop(EvolutionSetting(enabled=True), pool, evaluate_fn=None).run()


def test_loop_round0_only(tmp_path):
    """max_rounds=0 → 只跑 round 0。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=0)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate())
    result = loop.run(initial_directions=["d1", "d2"])
    assert result.rounds_completed == 1
    assert pool.size == 2
    assert all(e.operation == "original" for e in result.all_entries)


def test_loop_multi_rounds(tmp_path):
    """多轮: round 0 (original) + mutation + crossover。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=2, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate())
    result = loop.run(initial_directions=["d1", "d2"])
    assert result.rounds_completed == 2
    operations = [e.operation for e in result.all_entries]
    assert "original" in operations
    assert "mutation" in operations
    assert "crossover" in operations


def test_loop_records_parent_ids(tmp_path):
    """mutation/crossover 的 entry 含 parent_ids。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=2)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate())
    loop.run(initial_directions=["d1", "d2"])

    mutations = [e for e in pool.all() if e.operation == "mutation"]
    crossovers = [e for e in pool.all() if e.operation == "crossover"]
    assert all(len(m.parent_ids) == 1 for m in mutations)
    assert all(len(c.parent_ids) == 2 for c in crossovers)


def test_loop_lineage_chain(tmp_path):
    """mutation 子代可追溯到 original 父辈。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=1)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate())
    loop.run(initial_directions=["d1", "d2"])

    mutation = next(e for e in pool.all() if e.operation == "mutation")
    lineage = pool.lineage(mutation.entry_id)
    # lineage: original → mutation
    assert len(lineage) == 2
    assert lineage[0].operation == "original"
    assert lineage[1].entry_id == mutation.entry_id


def test_loop_early_stop(tmp_path):
    """early_stop_patience 触发后提前停止。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=10, early_stop_patience=2)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate(0.5))  # sharpe 恒定
    result = loop.run(initial_directions=["d1"])
    # round 0 必有 1 个 entry
    # round 1 选 parent 但 sharpe 不变, no_improve=1
    # round 2 选 parent 但 sharpe 不变, no_improve=2 → 停
    # 所以 rounds_completed <= 3
    assert result.rounds_completed <= 3


# ============================================================================
# 6. 与 QualityGate 集成 (2)
# ============================================================================

def test_loop_with_quality_gate_rejects(tmp_path):
    """quality_gate 拒绝的因子不进入 evaluate, 但记录到 pool。"""
    pool = TrajectoryPool(tmp_path)
    # 预填 Zoo 含 mock 的默认 expression, 让 round 0 必被 redundancy 拒绝
    mock_default_expr = "(close - close.shift(20)) / close.shift(20)"
    zoo = FactorZoo()
    zoo.add(mock_default_expr)
    gate = QualityGateNode(
        QualityGateSetting(redundancy=RedundancySetting(enabled=True, threshold=5)),
        zoo=zoo,
    )
    settings = EvolutionSetting(enabled=True, max_rounds=0, seed=42)
    loop = EvolutionLoop(settings, pool, quality_gate=gate, evaluate_fn=_make_evaluate())
    result = loop.run(initial_directions=["x"])

    # pool 应记录 rejected entry (redundancy 拦截)
    rejected = [e for e in pool.all() if not (e.feedback and e.feedback.decision)]
    assert len(rejected) >= 1
    assert result.rejected_count >= 1


def test_loop_with_quality_gate_passes(tmp_path):
    """quality_gate 通过时正常 evaluate。"""
    pool = TrajectoryPool(tmp_path)
    gate = QualityGateNode(QualityGateSetting())  # default
    settings = EvolutionSetting(enabled=True, max_rounds=0)
    loop = EvolutionLoop(settings, pool, quality_gate=gate, evaluate_fn=_make_evaluate())
    result = loop.run(initial_directions=["m"])
    assert result.total_count >= 1


# ============================================================================
# 7. 与 TrajectoryPool 集成 (1)
# ============================================================================

def test_loop_persists_to_pool(tmp_path):
    """结果持久化到 TrajectoryPool 双层 (Parquet + JSON)。"""
    pool = TrajectoryPool(tmp_path)
    settings = EvolutionSetting(enabled=True, max_rounds=1, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_make_evaluate())
    loop.run(initial_directions=["d1", "d2"])

    # 重载 pool, 验证持久化
    pool2 = TrajectoryPool(tmp_path)
    assert pool2.size == pool.size
    assert pool2.size >= 3  # 2 originals + 1 mutation/crossover
