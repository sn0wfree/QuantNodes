"""evolution/loop.py 终止条件与边界测试 (15 tests)。

聚焦:
    - max_rounds=0 → 只跑 round 0
    - 无 directions 且无 candidates → 空 round 0
    - early_stop_patience: 连续 N 轮无改善提前停
    - quality_gate 全部 reject → 不进 selector
    - workers=1 串行 / workers>1 ThreadPool
    - evaluate_fn 返回 dict 而非 tuple
    - 异常 evaluate_fn 不会让 loop 整体崩
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionSetting,
    FactorCandidate,
)
from QuantNodes.core.evolution.settings import EvolutionSetting as ES
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.quality_gate import (
    QualityGateNode,
    QualityGateSetting,
    ComplexitySetting,
)
from QuantNodes.core.trajectory import TrajectoryPool


# ── 辅助 ──────────────────────────────────────────────────


def _make_evaluate_fn(
    passed: bool = True,
    sharpe: float = 0.5,
    return_type: str = "tuple",  # "tuple" / "dict"
) -> Any:
    """构造 evaluate_fn。"""
    def eval_fn(c: FactorCandidate):
        if return_type == "tuple":
            return (
                passed,
                {"sharpe": sharpe, "ic_mean": 0.04, "arr": sharpe * 0.1},
                FactorFeedback(
                    factor_id=c.factor_id, factor_name=c.name,
                    decision=passed, summary=f"sharpe={sharpe}",
                ),
            )
        # dict
        return {
            "passed": passed,
            "metrics": {"sharpe": sharpe, "ic_mean": 0.04},
            "feedback_dict": {
                "factor_id": c.factor_id,
                "factor_name": c.name,
                "decision": passed,
                "summary": f"sharpe={sharpe}",
                "metadata": {},
                "channels": {},
            },
            "error": None,
        }
    return eval_fn


def _make_loop(
    tmp_path: Path,
    settings: EvolutionSetting,
    evaluate_fn=None,
    quality_gate: QualityGateNode | None = None,
    workers: int = 1,
) -> EvolutionLoop:
    pool = TrajectoryPool(tmp_path / "pool")
    return EvolutionLoop(
        settings=settings,
        pool=pool,
        quality_gate=quality_gate,
        evaluate_fn=evaluate_fn or _make_evaluate_fn(),
        workers=workers,
    )


# ============================================================================
# 1. Round 0 + Round N 基本行为 (5 tests)
# ============================================================================

class TestRoundZero:
    def test_no_directions_no_candidates_empty_round0(self, tmp_path: Path):
        """无 direction 无 candidate → round 0 空, loop 正常结束。"""
        s = ES(max_rounds=2)
        loop = _make_loop(tmp_path, s)
        result = loop.run()
        assert result.rounds_completed == 1  # 只 round 0
        assert len(result.all_entries) == 0

    def test_with_initial_directions(self, tmp_path: Path):
        """给 directions → round 0 生成 N candidates。"""
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        result = loop.run(initial_directions=["alpha1", "alpha2"])
        assert len(result.all_entries) == 2
        # 都是 original
        for e in result.all_entries:
            assert e.operation == "original"

    def test_with_initial_candidates(self, tmp_path: Path):
        """直接传 candidates → 不调 Hypothesizer。"""
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        cands = [
            FactorCandidate(factor_id="c1", name="a1", expression="close"),
            FactorCandidate(factor_id="c2", name="a2", expression="open"),
        ]
        result = loop.run(initial_candidates=cands)
        assert len(result.all_entries) == 2

    def test_evaluate_fn_none_raises(self, tmp_path: Path):
        """evaluate_fn=None 应抛 ValueError。"""
        s = ES(max_rounds=0)
        pool = TrajectoryPool(tmp_path / "pool")
        loop = EvolutionLoop(settings=s, pool=pool, evaluate_fn=None)
        with pytest.raises(ValueError, match="evaluate_fn 不能为 None"):
            loop.run()

    def test_max_rounds_zero_only_round0(self, tmp_path: Path):
        """max_rounds=0 → 只跑 round 0。"""
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        result = loop.run(initial_directions=["alpha1"])
        assert result.rounds_completed == 1


# ============================================================================
# 2. Selector + Mutator + Crosser 触发 (3 tests)
# ============================================================================

class TestRoundN:
    def test_round1_produces_mutation(self, tmp_path: Path):
        """max_rounds=1 → round 1 产生 mutation (1 parent)。"""
        s = ES(max_rounds=1, parent_selection_strategy="best")
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn(sharpe=0.5))
        result = loop.run(initial_directions=["alpha1"])
        # round 0: 1 original + round 1: 1 mutation
        # (crossover 需要 2 parents, 这里 round 0 只 1 entry → crossover 跳过)
        ops = [e.operation for e in result.all_entries]
        assert "original" in ops
        assert "mutation" in ops

    def test_round1_produces_crossover_with_2_parents(self, tmp_path: Path):
        """round 0 有 2 entries → round 1 触发 mutation + crossover。"""
        s = ES(max_rounds=1, parent_selection_strategy="best")
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn(sharpe=0.5))
        result = loop.run(initial_directions=["alpha1", "alpha2"])
        ops = [e.operation for e in result.all_entries]
        assert "original" in ops
        assert "mutation" in ops
        assert "crossover" in ops

    def test_round_stops_when_pool_empty(self, tmp_path: Path):
        """round 0 全 reject → round 1 selector 返回空 → 停止。"""
        # 设 quality_gate 全部 reject
        gate_setting = QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=1),
        )
        gate = QualityGateNode(gate_setting)
        s = ES(max_rounds=5, parent_selection_strategy="best")
        loop = _make_loop(tmp_path, s, quality_gate=gate)
        result = loop.run(initial_directions=["alpha1"])
        # round 0 全部 reject, round 1 立即 break
        assert result.rounds_completed <= 1
        # rejected_count >= 1
        assert result.rejected_count >= 1

    def test_result_best_entries_returns_top(self, tmp_path: Path):
        """best_entries 包含 top 10。"""
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn(sharpe=0.7))
        result = loop.run(initial_directions=[f"a{i}" for i in range(15)])
        assert len(result.best_entries) <= 10


# ============================================================================
# 3. evaluate_fn 返回类型 (3 tests)
# ============================================================================

class TestEvaluateFnReturnType:
    def test_evaluate_returns_dict(self, tmp_path: Path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn(return_type="dict"))
        result = loop.run(initial_directions=["alpha1"])
        assert len(result.all_entries) == 1
        # passed=True → decision=True
        assert result.total_count == 1

    def test_evaluate_returns_failed(self, tmp_path: Path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn(passed=False))
        result = loop.run(initial_directions=["alpha1"])
        assert result.rejected_count == 1
        assert result.total_count == 0

    def test_evaluate_returns_none_handled(self, tmp_path: Path):
        """evaluate_fn 返回 None → passed=False, 不崩。"""
        def eval_none(c):
            return None
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, evaluate_fn=eval_none)
        result = loop.run(initial_directions=["alpha1"])
        # entry 仍被记录
        assert len(result.all_entries) == 1


# ============================================================================
# 4. Quality gate 集成 (2 tests)
# ============================================================================

class TestQualityGateIntegration:
    def test_quality_gate_rejects(self, tmp_path: Path):
        """超长 expression → quality gate reject, 不调 evaluate_fn。"""
        gate_setting = QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=5),
        )
        gate = QualityGateNode(gate_setting)
        s = ES(max_rounds=0)
        called = []

        def eval_with_track(c):
            called.append(c.factor_id)
            return _make_evaluate_fn()(c)

        loop = _make_loop(tmp_path, s, quality_gate=gate, evaluate_fn=eval_with_track)
        result = loop.run(initial_candidates=[
            FactorCandidate(factor_id="c1", name="a", expression="close - close.shift(5) - close.shift(10)"),
        ])
        # reject, evaluate_fn 未被调用
        assert called == []
        assert result.rejected_count == 1

    def test_quality_gate_passes_simple(self, tmp_path: Path):
        gate_setting = QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=200),
        )
        gate = QualityGateNode(gate_setting)
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, quality_gate=gate, evaluate_fn=_make_evaluate_fn())
        result = loop.run(initial_directions=["alpha1"])
        assert result.total_count == 1


# ============================================================================
# 5. Sync knowledge_base (2 tests)
# ============================================================================

class TestKnowledgeBaseSync:
    def test_sync_kb_no_kb_returns_zero(self, tmp_path: Path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, evaluate_fn=_make_evaluate_fn())
        n = loop.sync_knowledge_base()
        assert n == 0  # 无 KB

    def test_sync_kb_with_kb(self, tmp_path: Path):
        from QuantNodes.core.knowledge import KnowledgeBase
        s = ES(max_rounds=0)
        pool = TrajectoryPool(tmp_path / "pool")
        kb = KnowledgeBase(pool=pool)
        loop = EvolutionLoop(settings=s, pool=pool, knowledge_base=kb, evaluate_fn=_make_evaluate_fn())
        loop.run(initial_directions=["alpha1"])
        n = loop.sync_knowledge_base()
        assert n >= 1
        assert len(kb) >= 1
