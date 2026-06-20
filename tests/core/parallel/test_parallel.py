"""parallel 模块测试 (Week 14) — 6 tests。

覆盖:
    - parallel_evaluate (3)
    - make_worker_evaluate (1)
    - EvolutionLoop workers=2 并行模式 (2)
"""
from __future__ import annotations

import tempfile
from pathlib import Path


from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting, FactorCandidate
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.parallel import (
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


# ============================================================================
# 4. ProcessPool mode (2)
# ============================================================================

def test_processpool_subprocess_evaluate():
    """ProcessPool subprocess_evaluate 运行 12 节点并返回结果。"""
    import pandas as pd
    import numpy as np
    from QuantNodes.core.parallel.worker_process import RunnerSnapshot, subprocess_evaluate

    n_days, n_stocks = 60, 20
    dates = [int(d.strftime('%Y%m%d')) for d in pd.bdate_range('2026-01-04', periods=n_days)]
    stocks = list(range(100001, 100001 + n_stocks))
    context = {
        'LoadData': {
            'factor': pd.DataFrame(np.random.randn(n_days, n_stocks), index=dates, columns=stocks),
            'price': pd.DataFrame(100 * np.exp(np.cumsum(np.random.randn(n_days, n_stocks) * 0.02, axis=0)), index=dates, columns=stocks),
            'id_citic1': pd.DataFrame(np.random.randint(1, 31, (n_days, n_stocks)), index=dates, columns=stocks),
            'mv_float': pd.DataFrame(np.random.lognormal(10, 1, (n_days, n_stocks)), index=dates, columns=stocks),
            'st': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'suspend': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'ud_limit': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'ipo_days': pd.DataFrame(np.ones((n_days, n_stocks), dtype=int) * 500, index=dates, columns=stocks),
            'index_cp': pd.DataFrame({'000300.SH': np.arange(3500, 3560), '000905.SH': np.arange(6000, 6060)}, index=dates),
            'stklist': pd.DataFrame(stocks, columns=[0]),
            'trade_dt': pd.DataFrame(dates, columns=[0]),
            '_loader': None,
        }
    }
    with tempfile.TemporaryDirectory() as td:
        snap = RunnerSnapshot(
            {'preprocess': {'adj_date_beg': 20260101, 'adj_date_end': 20260630, 'adj_mode': ['M', 'end']},
             'analysis': {'ic': {'min_group_size': 5}, 'group': {'groups': 5, 'factor_direction': 1, 'floor_mode': 'group', 'hedge': 'equal'}}},
            context
        )
        snap_path = Path(td) / 'snap.pkl'
        snap.save(snap_path)
        cand = FactorCandidate(factor_id='test', name='test_factor', expression='close - open')
        result = subprocess_evaluate(cand.__dict__, str(snap_path))
        assert result['passed'] is True
        assert result['error'] is None


def test_processpool_parallel_evaluate():
    """parallel_evaluate ProcessPool 模式并行评估。"""
    import pandas as pd
    import numpy as np
    from QuantNodes.core.parallel.worker_process import RunnerSnapshot
    from QuantNodes.core.parallel import parallel_evaluate

    n_days, n_stocks = 60, 20
    dates = [int(d.strftime('%Y%m%d')) for d in pd.bdate_range('2026-01-04', periods=n_days)]
    stocks = list(range(100001, 100001 + n_stocks))
    context = {
        'LoadData': {
            'factor': pd.DataFrame(np.random.randn(n_days, n_stocks), index=dates, columns=stocks),
            'price': pd.DataFrame(100 * np.exp(np.cumsum(np.random.randn(n_days, n_stocks) * 0.02, axis=0)), index=dates, columns=stocks),
            'id_citic1': pd.DataFrame(np.random.randint(1, 31, (n_days, n_stocks)), index=dates, columns=stocks),
            'mv_float': pd.DataFrame(np.random.lognormal(10, 1, (n_days, n_stocks)), index=dates, columns=stocks),
            'st': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'suspend': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'ud_limit': pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stocks),
            'ipo_days': pd.DataFrame(np.ones((n_days, n_stocks), dtype=int) * 500, index=dates, columns=stocks),
            'index_cp': pd.DataFrame({'000300.SH': np.arange(3500, 3560), '000905.SH': np.arange(6000, 6060)}, index=dates),
            'stklist': pd.DataFrame(stocks, columns=[0]),
            'trade_dt': pd.DataFrame(dates, columns=[0]),
            '_loader': None,
        }
    }
    with tempfile.TemporaryDirectory() as td:
        snap = RunnerSnapshot(
            {'preprocess': {'adj_date_beg': 20260101, 'adj_date_end': 20260630, 'adj_mode': ['M', 'end']},
             'analysis': {'ic': {'min_group_size': 5}, 'group': {'groups': 5, 'factor_direction': 1, 'floor_mode': 'group', 'hedge': 'equal'}}},
            context
        )
        snap_path = Path(td) / 'snap.pkl'
        snap.save(snap_path)

        cands = [FactorCandidate(factor_id=f'c{i}', name=f'factor_{i}', expression='close - open') for i in range(3)]
        results = parallel_evaluate(cands, _mock_eval, max_workers=2, snapshot_path=str(snap_path))
        assert len(results) == 3
        for r in results:
            assert r['passed'] is True


# ============================================================================
# C1+H3: 跨进程 hash / 种子确定性
# ============================================================================

class TestCrossProcessDeterminism:
    """验证 hash / 种子在多次启动中稳定。"""

    def test_zoo_hash_deterministic_across_calls(self):
        """同一 expression 多次 ast_hash 必返回相同 int。"""
        from QuantNodes.core.quality_gate.zoo import ast_hash
        h1 = ast_hash("close - close.shift(5)")
        h2 = ast_hash("close - close.shift(5)")
        h3 = ast_hash("close - close.shift(10)")  # 不同
        assert h1 == h2
        assert h1 != h3
        assert isinstance(h1, int)

    def test_zoo_hash_collision_resistant(self):
        """结构略不同的 expression 应 hash 不同。"""
        from QuantNodes.core.quality_gate.zoo import ast_hash
        h1 = ast_hash("close")
        h2 = ast_hash("open")
        h3 = ast_hash("high")
        assert len({h1, h2, h3}) == 3

    def test_heavy_evaluate_seed_deterministic(self):
        """_heavy_evaluate 同 expression 同 seed → 同 sharpe。"""
        from QuantNodes.core.parallel.worker import _heavy_evaluate
        c1 = {"factor_id": "e1", "name": "a", "expression": "close"}
        c2 = {"factor_id": "e1", "name": "a", "expression": "close"}
        r1 = _heavy_evaluate(c1)
        r2 = _heavy_evaluate(c2)
        assert abs(r1["metrics"]["sharpe"] - r2["metrics"]["sharpe"]) < 1e-9

    def test_heavy_evaluate_different_expr_different_metric(self):
        from QuantNodes.core.parallel.worker import _heavy_evaluate
        # 同 expression 跑 3 次, 必相同 (sha256 幂等)
        samples_close = [_heavy_evaluate({"expression": "close"})["metrics"]["sharpe"] for _ in range(3)]
        samples_open = [_heavy_evaluate({"expression": "open"})["metrics"]["sharpe"] for _ in range(3)]
        assert len(set(samples_close)) == 1  # close 3 次相同
        assert len(set(samples_open)) == 1   # open 3 次相同
        assert samples_close[0] != samples_open[0]  # close vs open 不同

    def test_heavy_evaluate_in_subprocess_deterministic(self):
        """真子进程跑 _heavy_evaluate, 结果与主进程一致。"""
        import subprocess
        import sys
        code = (
            "from QuantNodes.core.parallel.worker import _heavy_evaluate;"
            "r = _heavy_evaluate({'expression': 'close - close.shift(20)'});"
            "print(r['metrics']['sharpe'])"
        )
        # 跑 3 次
        results = []
        for _ in range(3):
            out = subprocess.check_output(
                [sys.executable, "-c", code],
                text=True, timeout=10,
            )
            results.append(float(out.strip()))
        # 3 次应完全相同 (sha256 跨进程幂等)
        assert results[0] == results[1] == results[2]

    def test_zoo_contains_across_subprocesses(self):
        """真子进程跑 contains, 与主进程一致。"""
        import subprocess
        import sys
        from QuantNodes.core.quality_gate.zoo import FactorZoo
        # 主进程 add
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            zoo_path = Path(td) / "zoo.parquet"
            zoo = FactorZoo(zoo_path)
            zoo.add("close - close.shift(5)")
            # 子进程: load + contains
            code = (
                "from QuantNodes.core.quality_gate.zoo import FactorZoo;"
                f"z = FactorZoo('{zoo_path}');"
                "print(z.contains('close - close.shift(5)'))"
            )
            out = subprocess.check_output(
                [sys.executable, "-c", code],
                text=True, timeout=10,
            )
            assert out.strip() == "True"
