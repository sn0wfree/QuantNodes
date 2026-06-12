"""EvolutionLoop + EvolutionSetting 全参数 parametrize (~25 tests)。

遍历 max_rounds/early_stop_patience/quality_gate 组合 + 异常。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from QuantNodes.core.evolution import (
    EvolutionLoop,
    FactorCandidate,
)
from QuantNodes.core.evolution.settings import EvolutionSetting as ES
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.quality_gate import (
    QualityGateNode,
    QualityGateSetting,
    ComplexitySetting,
    RedundancySetting,
)
from QuantNodes.core.trajectory import TrajectoryPool


def _make_evaluate_fn(passed: bool = True, sharpe: float = 0.5):
    def eval_fn(c: FactorCandidate):
        return (
            passed,
            {"sharpe": sharpe, "ic_mean": 0.04, "arr": sharpe * 0.1},
            FactorFeedback(
                factor_id=c.factor_id, factor_name=c.name,
                decision=passed, summary=f"sharpe={sharpe}",
            ),
        )
    return eval_fn


def _make_loop(tmp_path: Path, settings: ES, eval_fn=None, quality_gate=None, workers=1) -> EvolutionLoop:
    pool = TrajectoryPool(tmp_path / "pool")
    return EvolutionLoop(
        settings=settings, pool=pool,
        quality_gate=quality_gate,
        evaluate_fn=eval_fn or _make_evaluate_fn(),
        workers=workers,
    )


# ============================================================================
# 1. max_rounds 参数 (8 tests)
# ============================================================================

class TestMaxRounds:
    @pytest.mark.parametrize("max_rounds,expected_min_rounds", [
        (0, 1),  # 只 round 0
        (1, 1),
        (3, 2),
        (5, 3),
    ])
    def test_max_rounds_variants(self, tmp_path, max_rounds, expected_min_rounds):
        s = ES(max_rounds=max_rounds)
        loop = _make_loop(tmp_path, s, eval_fn=_make_evaluate_fn(sharpe=0.5))
        result = loop.run(initial_directions=["a", "b"])
        assert result.rounds_completed >= expected_min_rounds

    def test_max_rounds_zero_only_round0(self, tmp_path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        result = loop.run(initial_directions=["a"])
        assert result.rounds_completed == 1

    def test_max_rounds_with_no_directions(self, tmp_path):
        s = ES(max_rounds=5)
        loop = _make_loop(tmp_path, s)
        result = loop.run()
        # 无 directions/candidates → 只 round 0
        assert result.rounds_completed == 1

    @pytest.mark.parametrize("parents_per_round,n_mutations", [
        (1, 1),  # 默认
        (2, 2),  # 双 parent → 双 mutation child
        (3, 3),  # 三 parent → 三 mutation child
    ])
    def test_parents_per_round_produces_n_mutations(self, tmp_path, parents_per_round, n_mutations):
        """settings.parents_per_round 控制 mutation child 数量。"""
        s = ES(max_rounds=1, parents_per_round=parents_per_round)
        # 提供足够多 round 0 entry 让 selector 有 n 选
        loop = _make_loop(tmp_path, s, eval_fn=_make_evaluate_fn(sharpe=0.5))
        cands = [
            FactorCandidate(factor_id=f"c{i}", name=f"a{i}", expression="close")
            for i in range(5)
        ]
        result = loop.run(initial_candidates=cands)
        # round 0 5 entries + round 1 n mutation + 1 crossover
        mutations = [e for e in result.all_entries if e.operation == "mutation"]
        assert len(mutations) == n_mutations

    @pytest.mark.parametrize("top_n,expected_n", [
        (3, 3),
        (5, 5),
        (10, 10),
    ])
    def test_top_n_returns_n_best(self, tmp_path, top_n, expected_n):
        s = ES(max_rounds=0, top_n=top_n)
        loop = _make_loop(tmp_path, s)
        cands = [
            FactorCandidate(factor_id=f"c{i}", name=f"a{i}", expression="close")
            for i in range(20)
        ]
        result = loop.run(initial_candidates=cands)
        assert len(result.best_entries) == expected_n

    def test_max_rounds_quality_gate_rejects_all(self, tmp_path):
        """quality gate 全部 reject, round 1 立即 break。"""
        gate = QualityGateNode(QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=5),
        ))
        s = ES(max_rounds=10)
        loop = _make_loop(tmp_path, s, quality_gate=gate)
        result = loop.run(initial_directions=["a", "b", "c"])
        # round 0 全部 reject, round 1 selector 空 → break
        assert result.rounds_completed <= 1


# ============================================================================
# 2. initial_directions vs candidates (3 tests)
# ============================================================================

class TestInitialInput:
    def test_directions_only(self, tmp_path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        result = loop.run(initial_directions=["a", "b", "c"])
        assert len(result.all_entries) == 3

    def test_candidates_only(self, tmp_path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        cands = [FactorCandidate(factor_id=f"c{i}", name=f"a{i}", expression="close") for i in range(3)]
        result = loop.run(initial_candidates=cands)
        assert len(result.all_entries) == 3
        for e in result.all_entries:
            assert e.feedback.factor_id in {"c0", "c1", "c2"}

    def test_directions_plus_candidates(self, tmp_path):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s)
        cands = [FactorCandidate(factor_id="c1", name="a", expression="close")]
        result = loop.run(initial_directions=["d1", "d2"], initial_candidates=cands)
        assert len(result.all_entries) == 3


# ============================================================================
# 3. evaluate_fn 返回类型 (5 tests)
# ============================================================================

class TestEvaluateFnTypes:
    @pytest.mark.parametrize("return_value,expected_passed,description", [
        ((True, {"sharpe": 0.5}, None), True, "tuple"),
        ({"passed": True, "metrics": {"sharpe": 0.5}, "feedback_dict": {
            "factor_id": "x", "factor_name": "x", "decision": True, "summary": "",
            "metadata": {}, "channels": {},
        }, "error": None}, True, "dict"),
        (None, False, "None"),
        ("string", False, "string"),
        (42, False, "int"),
    ])
    def test_return_type_variants(self, tmp_path, return_value, expected_passed, description):
        s = ES(max_rounds=0)
        def eval_fn(c):
            return return_value
        loop = _make_loop(tmp_path, s, eval_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        # 1 entry created
        assert len(result.all_entries) == 1
        if expected_passed:
            assert result.total_count == 1
        else:
            assert result.rejected_count == 1


# ============================================================================
# 4. quality_gate 集成 (3 tests)
# ============================================================================

class TestQualityGateIntegration:
    @pytest.mark.parametrize("threshold,should_pass", [
        (5, False),  # 长度 > 5 必 reject
        (200, True),  # 长度 < 200 通过
    ])
    def test_complexity_threshold(self, tmp_path, threshold, should_pass):
        gate = QualityGateNode(QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=threshold),
        ))
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, quality_gate=gate)
        # 长 expression
        result = loop.run(initial_candidates=[
            FactorCandidate(factor_id="c1", name="a", expression="close - close.shift(5)"),
        ])
        if should_pass:
            assert result.total_count == 1
        else:
            assert result.rejected_count == 1

    def test_redundancy_check(self, tmp_path):
        """重复 expression 应被 reject。"""
        from QuantNodes.core.quality_gate import FactorZoo
        from QuantNodes.core.quality_gate.settings import RedundancySetting
        zoo = FactorZoo()
        zoo.add("close")  # 预存 close
        gate = QualityGateNode(QualityGateSetting(
            redundancy=RedundancySetting(enabled=True, threshold=5),
        ), zoo=zoo)
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, quality_gate=gate)
        # 提交 close → 应 reject (dist=0 < 5)
        result = loop.run(initial_candidates=[
            FactorCandidate(factor_id="c1", name="a", expression="close"),
        ])
        assert result.rejected_count == 1


# ============================================================================
# 5. workers 参数 (3 tests)
# ============================================================================

class TestWorkers:
    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test_workers_variants(self, tmp_path, workers):
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, workers=workers)
        result = loop.run(initial_directions=["a", "b", "c"])
        # 全部 eval
        assert len(result.all_entries) == 3

    def test_workers_zero_or_negative_falls_back(self, tmp_path):
        s = ES(max_rounds=0)
        # workers=0 应走串行
        loop = _make_loop(tmp_path, s, workers=0)
        result = loop.run(initial_directions=["a"])
        assert result.rounds_completed == 1


# ============================================================================
# 6. 异常 (3 tests)
# ============================================================================

class TestErrors:
    def test_evaluate_fn_none_raises(self, tmp_path):
        s = ES(max_rounds=0)
        pool = TrajectoryPool(tmp_path / "pool")
        loop = EvolutionLoop(settings=s, pool=pool, evaluate_fn=None)
        with pytest.raises(ValueError, match="evaluate_fn 不能为 None"):
            loop.run(initial_directions=["a"])

    def test_evaluate_fn_runtime_error_propagates(self, tmp_path):
        """evaluate_fn RuntimeError 当前未捕获, 异常上抛。"""
        def bad(c):
            raise RuntimeError("bad")
        s = ES(max_rounds=0)
        loop = _make_loop(tmp_path, s, eval_fn=bad)
        with pytest.raises(RuntimeError, match="bad"):
            loop.run(initial_directions=["a"])
