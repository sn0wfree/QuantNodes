"""演化框架 E2E 测试: 完整 3 轮演化 (hypothesize → mutate → crossover) + quality_gate。

不在 PipelineRunner.run_evolution() 中跑 (需要真实 H5 数据),
而是直接用 EvolutionLoop + TrajectoryPool + QualityGateNode,
通过 mock evaluate_fn 模拟 12 节点回测, 验证:
    - 3 轮演化 (round 0 + mutation + crossover)
    - 谱系可追溯
    - quality_gate REJECTED 不进入 evaluate
    - Parquet + JSON 双层持久化 + 重载
    - CLI factor-info / factor-best 显示正确
"""
from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from QuantNodes.cli import cmd_factor_best, cmd_factor_info
from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionSetting,
    FactorCandidate,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.quality_gate import (
    FactorZoo,
    QualityGateNode,
    QualityGateSetting,
    RedundancySetting,
)
from QuantNodes.core.trajectory import (
    SelectionStrategy,
    TrajectoryEntry,
    TrajectoryPool,
)


# ============================================================================
# 1. 完整 3 轮演化 (round 0 → mutation → crossover)
# ============================================================================

def _mock_evaluate(metric_value: float = 0.5, passed: bool = True) -> Callable:
    """模拟 12 节点回测, 返回固定 metric + FactorFeedback。"""
    def evaluate(candidate: FactorCandidate) -> tuple[bool, dict, FactorFeedback]:
        return (
            passed,
            {"sharpe": metric_value, "arr": metric_value * 0.1, "ic_mean": 0.04},
            FactorFeedback(
                factor_id=candidate.factor_id,
                factor_name=candidate.name,
                decision=passed,
                summary=f"sharpe={metric_value:.2f}",
            ),
        )
    return evaluate


def test_e2e_three_rounds_full_chain(tmp_path):
    """完整 3 轮演化: round 0 (3 originals) → mutation → crossover。"""
    pool = TrajectoryPool(tmp_path / "pool")
    settings = EvolutionSetting(
        enabled=True,
        max_rounds=2,  # round 1 (mutation) + round 2 (crossover)
        parent_selection_strategy="top_percent_plus_random",
        top_percent_threshold=0.5,
        seed=42,
    )
    loop = EvolutionLoop(
        settings,
        pool=pool,
        evaluate_fn=_mock_evaluate(0.5),
    )
    result = loop.run(initial_directions=["momentum", "reversal", "volatility"])

    # 3 round 0 entries + 1 mutation + 1 crossover = 5
    assert result.rounds_completed == 2
    assert pool.size == 5

    # 验证每种 operation 存在
    ops = [e.operation for e in pool.all()]
    assert ops.count("original") == 3
    assert ops.count("mutation") == 1
    assert ops.count("crossover") == 1


def test_e2e_lineage_traceable(tmp_path):
    """E2E: 谱系可追溯 (mutation 1 parent, crossover 2 parents)。"""
    pool = TrajectoryPool(tmp_path / "pool")
    settings = EvolutionSetting(enabled=True, max_rounds=2, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_mock_evaluate(0.5))
    loop.run(initial_directions=["d1", "d2"])

    mutation = next(e for e in pool.all() if e.operation == "mutation")
    crossover = next(e for e in pool.all() if e.operation == "crossover")

    # mutation 1 parent
    assert len(mutation.parent_ids) == 1
    # crossover 2 parents
    assert len(crossover.parent_ids) == 2

    # 谱系回溯: mutation 父辈是某个 original
    mut_lineage = pool.lineage(mutation.entry_id)
    assert len(mut_lineage) == 2
    assert mut_lineage[0].operation == "original"
    assert mut_lineage[1].entry_id == mutation.entry_id

    # crossover 谱系: original → crossover (走第一 parent)
    cross_lineage = pool.lineage(crossover.entry_id)
    assert cross_lineage[-1].entry_id == crossover.entry_id
    assert cross_lineage[-2].operation in ("original", "mutation")


def test_e2e_metrics_accumulate(tmp_path):
    """E2E: metrics 正确传递并排序。"""
    pool = TrajectoryPool(tmp_path / "pool")
    settings = EvolutionSetting(enabled=True, max_rounds=1, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_mock_evaluate(0.7))
    loop.run(initial_directions=["d1", "d2"])

    for e in pool.all():
        assert e.metrics.get("sharpe") == 0.7
        assert e.feedback.decision is True

    # best() 排序
    top = pool.best(top_n=2, metric="sharpe")
    assert all(e.metrics["sharpe"] == 0.7 for e in top)


# ============================================================================
# 2. QualityGate REJECTED 不进入 evaluate
# ============================================================================

def test_e2e_quality_gate_rejects_rejected_entries(tmp_path):
    """E2E: quality_gate 拒绝的因子不调 evaluate_fn, 但写入 pool。"""
    pool = TrajectoryPool(tmp_path / "pool")
    # 预填 Zoo 含 mock 默认 expression, 让 round 0 全被拒
    mock_default_expr = "(close - close.shift(20)) / close.shift(20)"
    zoo = FactorZoo()
    zoo.add(mock_default_expr)
    gate = QualityGateNode(
        QualityGateSetting(redundancy=RedundancySetting(enabled=True, threshold=5)),
        zoo=zoo,
    )

    evaluate_calls: list[str] = []

    def tracking_evaluate(candidate: FactorCandidate) -> tuple[bool, dict, FactorFeedback]:
        evaluate_calls.append(candidate.factor_id)
        return (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=candidate.factor_id, factor_name=candidate.name,
            decision=True, summary="ok",
        ))

    settings = EvolutionSetting(enabled=True, max_rounds=0, seed=42)
    loop = EvolutionLoop(settings, pool, quality_gate=gate, evaluate_fn=tracking_evaluate)
    result = loop.run(initial_directions=["x", "y"])

    # round 0 的 2 个 candidate 全被 quality_gate 拦截 → evaluate_fn 0 次调用
    assert len(evaluate_calls) == 0
    # 但 pool 仍记录 2 个 rejected entry
    assert pool.size == 2
    rejected = [e for e in pool.all() if not e.feedback.decision]
    assert len(rejected) == 2
    assert result.rejected_count == 2
    assert result.total_count == 0


def test_e2e_quality_gate_passes_through(tmp_path):
    """E2E: quality_gate 通过时调 evaluate_fn。"""
    pool = TrajectoryPool(tmp_path / "pool")
    gate = QualityGateNode(QualityGateSetting())  # 空 zoo → 全通过

    evaluate_calls: list[str] = []

    def tracking_evaluate(candidate: FactorCandidate) -> tuple[bool, dict, FactorFeedback]:
        evaluate_calls.append(candidate.factor_id)
        return (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=candidate.factor_id, factor_name=candidate.name,
            decision=True, summary="ok",
        ))

    settings = EvolutionSetting(enabled=True, max_rounds=0, seed=42)
    loop = EvolutionLoop(settings, pool, quality_gate=gate, evaluate_fn=tracking_evaluate)
    result = loop.run(initial_directions=["momentum", "reversal"])

    assert len(evaluate_calls) == 2
    assert result.total_count == 2
    assert result.rejected_count == 0


# ============================================================================
# 3. 持久化 + 重载
# ============================================================================

def test_e2e_persist_and_reload(tmp_path):
    """E2E: pool 双层持久化 (Parquet + JSON) + 重载一致。"""
    base = tmp_path / "pool"
    pool = TrajectoryPool(base)
    settings = EvolutionSetting(enabled=True, max_rounds=1, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_mock_evaluate(0.5))
    loop.run(initial_directions=["d1", "d2"])
    original_size = pool.size

    # 重载
    pool2 = TrajectoryPool(base)
    assert pool2.size == original_size

    # Parquet 文件存在
    assert (base / "trajectories.parquet").exists()
    # JSON 文件存在 (每条 entry 一个)
    json_files = list(base.glob("*.json"))
    assert len(json_files) == original_size


def test_e2e_parquet_schema(tmp_path):
    """E2E: Parquet schema 包含 15 列。"""
    base = tmp_path / "pool"
    pool = TrajectoryPool(base)
    settings = EvolutionSetting(enabled=True, max_rounds=0, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_mock_evaluate(0.6))
    loop.run(initial_directions=["d1"])

    df = pd.read_parquet(base / "trajectories.parquet")
    expected_cols = {
        "entry_id", "round_idx", "operation", "parent_ids",
        "decision", "duration_ms", "timestamp", "factor_name", "summary",
        "ic_mean", "rank_ic_mean", "sharpe", "arr", "mdd", "calmar",
    }
    assert set(df.columns) == expected_cols
    assert df.iloc[0]["sharpe"] == 0.6


# ============================================================================
# 4. CLI 集成: factor-info / factor-best
# ============================================================================

def test_e2e_cli_factor_info_after_evolution(tmp_path):
    """E2E: 演化完成后, CLI factor-info 正确显示统计。"""
    base = tmp_path / "pool"
    pool = TrajectoryPool(base)
    settings = EvolutionSetting(enabled=True, max_rounds=2, seed=42)
    loop = EvolutionLoop(settings, pool, evaluate_fn=_mock_evaluate(0.5))
    loop.run(initial_directions=["d1", "d2"])

    class Args:
        pool_dir = str(base)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_info(Args())
    assert rc == 0
    out = buf.getvalue()
    assert "size: 4" in out  # 2 originals + 1 mutation + 1 crossover
    assert "by_round:" in out
    assert "by_operation: {'original': 2, 'mutation': 1, 'crossover': 1}" in out


def test_e2e_cli_factor_best_after_evolution(tmp_path):
    """E2E: 演化完成后, CLI factor-best 正确排序。"""
    base = tmp_path / "pool"

    # 构造不同 sharpe 的 entry
    pool = TrajectoryPool(base)
    pool.add(TrajectoryEntry(
        entry_id="e1",
        round_idx=0, operation="original",
        feedback=FactorFeedback(factor_name="low", decision=True, summary="ok"),
        metrics={"sharpe": 0.3},
    ))
    pool.add(TrajectoryEntry(
        entry_id="e2",
        round_idx=1, operation="mutation",
        feedback=FactorFeedback(factor_name="high", decision=True, summary="ok"),
        metrics={"sharpe": 1.5},
    ))

    class Args:
        pool_dir = str(base)
        top = 5
        metric = "sharpe"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_best(Args())
    assert rc == 0
    out = buf.getvalue()
    pos_high = out.find("high [")
    pos_low = out.find("low [")
    assert pos_high < pos_low


# ============================================================================
# 5. 端到端: CLI → YAML → runner → pool → CLI
# ============================================================================

def test_e2e_yaml_to_cli_full_flow(tmp_path):
    """E2E: YAML 配置 → PipelineRunner → EvolutionLoop → pool → CLI factor-info。"""
    import yaml
    base = tmp_path / "pool"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "factor": {"name": "test", "factor_dir": "x.h5"},
        "preprocess": {
            "adj_date_beg": 20240101,
            "adj_date_end": 20240301,
        },
        "evolution": {
            "enabled": True,
            "max_rounds": 1,
            "pool_dir": str(base),
        },
    }))

    from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
    runner = PipelineRunner.from_yaml(str(config_path))
    # 注入 mock evaluate_fn, 避免真实回测
    pool = runner._build_trajectory_pool()
    from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting as _ES
    loop = EvolutionLoop(
        _ES(enabled=True, max_rounds=1, seed=42),
        pool=pool,
        evaluate_fn=_mock_evaluate(0.5),
    )
    result = loop.run(initial_directions=["d1", "d2"])

    assert pool.size >= 3

    # CLI 验证
    class Args:
        pool_dir = str(base)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_factor_info(Args())
    out = buf.getvalue()
    assert "size:" in out
    assert "by_operation:" in out
