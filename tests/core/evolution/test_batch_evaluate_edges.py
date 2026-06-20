"""EvolutionLoop._batch_evaluate_and_record 错误处理测试 (10 tests)。

通过直接调用 _batch_evaluate_and_record 验证错误处理。

聚焦:
    - evaluate_fn 返回 tuple (passed, metrics, feedback) — 默认路径
    - evaluate_fn 返回 dict 含 passed/metrics/feedback_dict — 替代路径
    - evaluate_fn 返回 None — 兜底
    - evaluate_fn 返回其他类型 (str, int) — 兜底 passed=False
    - quality_gate 全部 reject 时跳过 evaluate_fn
    - 混合 passed/未 passed 候选
    - 缺 feedback_dict 时 metadata 从 metrics 取
"""
from __future__ import annotations

from pathlib import Path


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


def _make_loop(
    tmp_path: Path,
    settings: EvolutionSetting,
    evaluate_fn=None,
    quality_gate: QualityGateNode | None = None,
) -> EvolutionLoop:
    pool = TrajectoryPool(tmp_path / "pool")
    return EvolutionLoop(
        settings=settings,
        pool=pool,
        quality_gate=quality_gate,
        evaluate_fn=evaluate_fn or (lambda c: (True, {"sharpe": 0.5}, None)),
    )


def _cand(factor_id: str, name: str = "f", expression: str = "close") -> FactorCandidate:
    return FactorCandidate(
        factor_id=factor_id, name=name, expression=expression,
    )


# ============================================================================
# 1. evaluate_fn 返回类型 (5 tests)
# ============================================================================

class TestEvaluateFnReturnTypes:
    def test_tuple_with_feedback(self, tmp_path: Path):
        """tuple (passed, metrics, feedback) 标准路径。"""
        def eval_fn(c):
            return (
                True,
                {"sharpe": 0.5, "ic_mean": 0.04},
                FactorFeedback(
                    factor_id=c.factor_id, factor_name=c.name,
                    decision=True, summary="ok",
                ),
            )
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        assert result.total_count == 1

    def test_dict_with_feedback_dict(self, tmp_path: Path):
        """dict 含 passed/metrics/feedback_dict。"""
        def eval_fn(c):
            return {
                "passed": True,
                "metrics": {"sharpe": 1.0},
                "feedback_dict": {
                    "factor_id": c.factor_id,
                    "factor_name": c.name,
                    "decision": True,
                    "summary": "from dict",
                    "metadata": {},
                    "channels": {},
                },
                "error": None,
            }
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        assert result.total_count == 1

    def test_returns_none(self, tmp_path: Path):
        """evaluate_fn 返回 None → passed=False, 不崩。"""
        def eval_fn(c):
            return None
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        # 1 entry, passed=False
        assert result.rejected_count == 1
        assert result.total_count == 0

    def test_returns_string(self, tmp_path: Path):
        """evaluate_fn 返回 str → 兜底 passed=False。"""
        def eval_fn(c):
            return "some string"
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        assert result.rejected_count == 1

    def test_returns_int(self, tmp_path: Path):
        def eval_fn(c):
            return 42
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        assert result.rejected_count == 1


# ============================================================================
# 2. Quality gate 短路 (2 tests)
# ============================================================================

class TestQualityGateShortCircuit:
    def test_quality_gate_rejects_skips_eval(self, tmp_path: Path):
        """quality gate reject → 不调 evaluate_fn。"""
        gate = QualityGateNode(QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=5),
        ))
        called = []
        def eval_fn(c):
            called.append(c.factor_id)
            return (True, {"sharpe": 0.5}, None)
        loop = _make_loop(tmp_path, ES(max_rounds=0), quality_gate=gate, evaluate_fn=eval_fn)
        # 长 expression → reject
        result = loop.run(initial_candidates=[
            _cand("c1", expression="close - close.shift(5) - close.shift(10) - close.shift(20)"),
        ])
        # eval_fn 未被调用
        assert called == []
        assert result.rejected_count == 1

    def test_quality_gate_passes_runs_eval(self, tmp_path: Path):
        """quality gate pass → eval_fn 被调用。"""
        gate = QualityGateNode(QualityGateSetting(
            complexity=ComplexitySetting(enabled=True, symbol_length_threshold=200),
        ))
        called = []
        def eval_fn(c):
            called.append(c.factor_id)
            return (True, {"sharpe": 0.5}, None)
        loop = _make_loop(tmp_path, ES(max_rounds=0), quality_gate=gate, evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["alpha"])
        # 1 entry, factor_id 是 UUID
        assert len(called) == 1
        assert result.total_count == 1


# ============================================================================
# 3. 混合 candidates (3 tests)
# ============================================================================

class TestMixedCandidates:
    def test_mixed_pass_fail_results(self, tmp_path: Path):
        """部分 passed, 部分 failed。"""
        idx = [0]
        def eval_fn(c):
            idx[0] += 1
            return (idx[0] % 2 == 0, {"sharpe": 0.5}, None)
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a", "b", "c", "d"])
        # 2 passed (idx 2, 4), 2 failed
        assert result.total_count + result.rejected_count == 4
        assert result.total_count >= 1
        assert result.rejected_count >= 1

    def test_metrics_extracted_to_entry(self, tmp_path: Path):
        """metrics 应被提取到 entry.metrics。"""
        def eval_fn(c):
            return (
                True,
                {"sharpe": 1.5, "ic_mean": 0.05, "extra": "x"},
                FactorFeedback(
                    factor_id=c.factor_id, factor_name=c.name,
                    decision=True, summary="ok",
                ),
            )
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a"])
        entry = result.all_entries[0]
        assert entry.metrics["sharpe"] == 1.5
        assert entry.metrics["ic_mean"] == 0.05
        # extra 也保留
        assert entry.metrics["extra"] == "x"

    def test_pool_size_after_run(self, tmp_path: Path):
        """run 后 pool 包含所有 entry (passed + failed)。"""
        def eval_fn(c):
            return (False, {}, None)
        loop = _make_loop(tmp_path, ES(max_rounds=0), evaluate_fn=eval_fn)
        result = loop.run(initial_directions=["a", "b", "c"])
        # pool 包含 3 个 rejected
        assert loop.pool.size == 3
        assert result.rejected_count == 3
