"""parallel 模块测试 (Week 14) — 6 tests。

覆盖:
    - parallel_evaluate (3)
    - make_worker_evaluate (1)
    - EvolutionLoop workers=2 并行模式 (2)
"""
from __future__ import annotations

import tempfile

import pytest

from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting, FactorCandidate
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.parallel import (
    detect_max_workers,
    make_worker_evaluate,
    parallel_evaluate,
)
from QuantNodes.core.trajectory import TrajectoryPool


# ============================================================================
# Helpers
# ============================================================================

def _mock_eval(c):
    return (True, {"sharpe": 0.5}, FactorFeedback(
        factor_id=c.factor_id, factor_name=c.name,
        decision=True, summary="ok",
    ))


# ============================================================================
# 1. parallel_evaluate (3)
# ============================================================================

def test_parallel_evaluate_serial():
    """workers=1 串行。"""
    candidates = [FactorCandidate(factor_id=f"i{i}", name=f"n{i}", expression=f"expr_{i}") for i in range(3)]
    results = parallel_evaluate(candidates, _mock_eval, max_workers=1)
    assert len(results) == 3
    assert all(isinstance(r, tuple) for r in results)


def test_parallel_evaluate_threaded():
    """workers=4 ThreadPool 并行。"""
    candidates = [FactorCandidate(factor_id=f"i{i}", name=f"n{i}", expression=f"expr_{i}") for i in range(5)]
    results = parallel_evaluate(candidates, _mock_eval, max_workers=4)
    assert len(results) == 5
    for r in results:
        assert isinstance(r, tuple)
        assert r[0] is True


def test_parallel_evaluate_empty():
    """空 list → 空结果。"""
    results = parallel_evaluate([], _mock_eval, max_workers=1)
    assert results == []


# ============================================================================
# 2. make_worker_evaluate (1)
# ============================================================================

def test_make_worker_evaluate():
    """make_worker_evaluate 包装函数可调用。"""
    def my_eval(c):
        return (True, {"sharpe": 1.0}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name,
            decision=True, summary="ok",
        ))
    worker_fn = make_worker_evaluate(my_eval)
    result = worker_fn(FactorCandidate(factor_id="x", name="y", expression="expr"))
    assert result[0] is True
    assert result[1]["sharpe"] == 1.0


# ============================================================================
# 3. EvolutionLoop workers=2 (2)
# ============================================================================

def test_workers2_generates_more_entries(tmp_path):
    """workers=2 每轮产生 mutation + crossover, entries 比 workers=1 多。"""
    with tempfile.TemporaryDirectory() as td:
        pool1 = TrajectoryPool(td)
        settings = EvolutionSetting(enabled=True, max_rounds=2, seed=42)
        loop1 = EvolutionLoop(settings, pool=pool1, evaluate_fn=_mock_eval, workers=1)
        loop1.run(initial_directions=["d1", "d2"])

    with tempfile.TemporaryDirectory() as td2:
        pool2 = TrajectoryPool(td2)
        settings2 = EvolutionSetting(enabled=True, max_rounds=2, seed=42)
        loop2 = EvolutionLoop(settings2, pool=pool2, evaluate_fn=_mock_eval, workers=2)
        loop2.run(initial_directions=["d1", "d2"])

    assert pool2.size >= pool1.size
    # workers=2 应产生 crossover
    ops2 = [e.operation for e in pool2.all()]
    assert "crossover" in ops2


def test_workers2_metrics_correct(tmp_path):
    """workers=2 结果 metric 正确。"""
    with tempfile.TemporaryDirectory() as td:
        pool = TrajectoryPool(td)
        settings = EvolutionSetting(enabled=True, max_rounds=1, seed=42)
        loop = EvolutionLoop(settings, pool=pool, evaluate_fn=_mock_eval, workers=2)
        result = loop.run(initial_directions=["d1", "d2"])

    assert result.total_count >= 2
    assert all(
        (e.metrics or {}).get("sharpe", 0) == 0.5
        for e in pool.all() if e.feedback and e.feedback.decision
    )
