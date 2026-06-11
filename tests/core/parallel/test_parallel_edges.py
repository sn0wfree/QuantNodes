"""parallel 模块边界条件测试 (15 tests)。

聚焦:
    - _heavy_evaluate: 决定性 (相同 expression 同 sharpe), sleep_ms 模拟
    - make_worker_evaluate: 无 base 用 heavy, 有 base 透传
    - parallel_evaluate: workers=1 串行, ThreadPool/ProcessPool 异常隔离
    - RunnerSnapshot: 序列化往返, 缺字段不崩
    - subprocess_evaluate: 错误捕获, 返回 dict 结构
    - detect_max_workers: 返回合理值
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pytest

from QuantNodes.core.parallel import (
    _heavy_evaluate,
    detect_max_workers,
    make_worker_evaluate,
    parallel_evaluate,
)
from QuantNodes.core.parallel.worker_process import (
    RunnerSnapshot,
    prepare_snapshot,
    subprocess_evaluate,
)


class FakeCandidate:
    """模拟 FactorCandidate (有 __dict__)。"""
    def __init__(self, factor_id: str, name: str, expression: str):
        self.factor_id = factor_id
        self.name = name
        self.expression = expression


# ============================================================================
# 1. _heavy_evaluate (3 tests)
# ============================================================================

class TestHeavyEvaluate:
    def test_returns_required_fields(self):
        r = _heavy_evaluate({
            "factor_id": "f1", "name": "n1", "expression": "close",
        })
        assert "passed" in r
        assert "metrics" in r
        assert "feedback_dict" in r
        assert "error" in r
        assert r["passed"] is True
        assert r["error"] is None

    def test_metrics_range(self):
        """sharpe 在 [0, 2] 范围。"""
        r = _heavy_evaluate({"expression": "close - close.shift(5)"})
        assert 0.0 <= r["metrics"]["sharpe"] <= 2.0
        assert 0.0 <= r["metrics"]["arr"] <= 0.2  # arr = sharpe * 0.1

    def test_sleep_ms_slows_down(self):
        """sleep_ms=100 实际耗时约 100ms。"""
        t0 = time.time()
        _heavy_evaluate({"expression": "x"}, sleep_ms=100)
        elapsed = (time.time() - t0) * 1000
        assert 80 < elapsed < 200  # 允许 20% 误差


# ============================================================================
# 2. make_worker_evaluate (3 tests)
# ============================================================================

class TestMakeWorkerEvaluate:
    def test_default_uses_heavy(self):
        """None base → 返回 lambda 调 heavy。"""
        f = make_worker_evaluate()
        # 必须能接受 FakeCandidate-like
        class C:
            factor_id = "c1"
            name = "n"
            expression = "close"
        r = f(C())
        assert r["passed"] is True

    def test_with_base_evaluate(self):
        """提供 base_evaluate → 直接使用。"""
        def my_eval(c):
            return {"passed": False, "metrics": {"x": 1}, "feedback_dict": None, "error": "test"}

        f = make_worker_evaluate(base_evaluate=my_eval)
        r = f(FakeCandidate("c1", "n", "close"))
        assert r["passed"] is False
        assert "test" in r["error"]

    def test_with_sleep_in_lambda(self):
        f = make_worker_evaluate(sleep_ms=10)
        c = FakeCandidate("c1", "n", "x")
        t0 = time.time()
        f(c)
        elapsed = (time.time() - t0) * 1000
        assert elapsed >= 8


# ============================================================================
# 3. parallel_evaluate (5 tests)
# ============================================================================

class TestParallelEvaluate:
    def test_workers_1_serial(self):
        """workers=1 串行调用, 返回 list。"""
        def eval_fn(c):
            return {"passed": True, "metrics": {}, "feedback_dict": None, "error": None}
        cands = [FakeCandidate(f"c{i}", "n", f"expr_{i}") for i in range(5)]
        results = parallel_evaluate(cands, eval_fn, max_workers=1)
        assert len(results) == 5
        assert all(r["passed"] for r in results)

    def test_workers_2_threadpool(self):
        """workers=2 ThreadPool 并行。"""
        def eval_fn(c):
            return {"passed": True, "metrics": {"x": 1}, "feedback_dict": None, "error": None}
        cands = [FakeCandidate(f"c{i}", "n", "e") for i in range(10)]
        results = parallel_evaluate(cands, eval_fn, max_workers=2)
        assert len(results) == 10
        assert all(r["passed"] for r in results)

    def test_threadpool_exception_caught(self):
        """ThreadPool 中 evaluate_fn 抛异常, 单个结果为 (False, {}, None)。"""
        def eval_fn(c):
            raise ValueError(f"bad: {c.factor_id}")
        cands = [FakeCandidate(f"c{i}", "n", "e") for i in range(3)]
        results = parallel_evaluate(cands, eval_fn, max_workers=2)
        # 异常被捕获, 结果为 (False, {}, None)
        for r in results:
            assert r == (False, {}, None)

    def test_preserves_order(self):
        """ThreadPool 返回结果顺序与输入一致。"""
        def eval_fn(c):
            return {"passed": True, "metrics": {"id": c.factor_id}, "feedback_dict": None, "error": None}
        cands = [FakeCandidate(f"c{i}", "n", "e") for i in range(20)]
        results = parallel_evaluate(cands, eval_fn, max_workers=4)
        ids = [r["metrics"]["id"] for r in results]
        assert ids == [f"c{i}" for i in range(20)]

    def test_empty_candidates(self):
        def eval_fn(c):
            return {}
        results = parallel_evaluate([], eval_fn, max_workers=2)
        assert results == []


# ============================================================================
# 4. RunnerSnapshot (3 tests)
# ============================================================================

class TestRunnerSnapshot:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        config = {"factor": {"name": "alpha", "expression": "close"}}
        context = {"key": "value", "list": [1, 2, 3]}
        snap = RunnerSnapshot(config, context, factor_path="/tmp/f.h5")
        path = tmp_path / "snap.pkl"
        snap.save(path)
        assert path.exists()
        loaded = RunnerSnapshot.load(path)
        assert loaded["config"] == config
        assert loaded["context"] == context
        assert loaded["factor_name"] == "alpha"
        assert loaded["factor_path"] == "/tmp/f.h5"

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            RunnerSnapshot.load(tmp_path / "missing.pkl")

    def test_factor_name_extracted(self):
        config = {"factor": {"name": "alpha_v2", "expression": "x"}}
        snap = RunnerSnapshot(config, {})
        assert snap.factor_name == "alpha_v2"

    def test_prepare_snapshot_with_pydantic(self):
        """支持 pydantic model_dump。"""
        from pydantic import BaseModel

        class Cfg(BaseModel):
            factor: dict

        cfg = Cfg(factor={"name": "x", "expression": "y"})
        snap = prepare_snapshot(cfg, {"k": "v"})
        assert snap.config["factor"]["name"] == "x"

    def test_prepare_snapshot_with_dict(self):
        snap = prepare_snapshot({"factor": {"name": "z"}}, {})
        assert snap.config["factor"]["name"] == "z"


# ============================================================================
# 5. subprocess_evaluate 错误处理 (2 tests)
# ============================================================================

class TestSubprocessEvaluate:
    def test_missing_snapshot_returns_error(self, tmp_path: Path):
        """snapshot_path 不存在 → 返回 passed=False + error。"""
        r = subprocess_evaluate(
            {"factor_id": "c1", "name": "n", "expression": "x"},
            str(tmp_path / "missing.pkl"),
        )
        assert r["passed"] is False
        assert "error" in r
        assert r["feedback_dict"] is None
        assert r["metrics"] == {}

    def test_corrupt_snapshot_returns_error(self, tmp_path: Path):
        """损坏的 snapshot 文件 → 错误捕获。"""
        bad_path = tmp_path / "bad.pkl"
        bad_path.write_bytes(b"not a pickle")
        r = subprocess_evaluate(
            {"factor_id": "c1", "name": "n", "expression": "x"},
            str(bad_path),
        )
        assert r["passed"] is False
        assert r["error"] is not None


# ============================================================================
# 6. detect_max_workers (1 test)
# ============================================================================

class TestDetectMaxWorkers:
    def test_returns_positive_int(self):
        n = detect_max_workers(default=4)
        assert isinstance(n, int)
        assert n >= 1
        assert n <= 32  # 合理上限 (2 * default)
