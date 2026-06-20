# coding: utf-8
"""演化适配器 / Evolution Adapter.

把 ``PipelineRunner`` 与 ``QuantNodes.core.evolution`` 框架对接:

- 构造 ``QualityGateNode`` (按 ``cfg.quality_gate.enabled``)
- 构造 ``TrajectoryPool`` (按 ``cfg.evolution.enabled``)
- 构造 ``EvolutionLoop`` (含 evaluate_fn)
- ``evaluate_candidate`` / ``run_one_factor`` 子例程
- ``run_evolution`` 主入口 (含 ProcessPool 快照预序列化 + Streaming 集成)

Phase R2 (2026-06-19): 从 pipeline_runner.py 抽出, 单一职责.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionResult,
    EvolutionSetting,
    FactorCandidate,
)
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.monitoring import MetricCollector, generate_dashboard_html
from QuantNodes.core.parallel.worker_process import prepare_snapshot
from QuantNodes.core.quality_gate import (
    FactorZoo,
    QualityGateNode,
    QualityGateSetting,
)
from QuantNodes.core.trajectory import TrajectoryPool
from QuantNodes.research.factor_test.utils.metrics_extractor import (
    extract_metrics_from_ctx,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner


def build_quality_gate(runner: "PipelineRunner") -> Optional[QualityGateNode]:
    """根据 ``cfg.quality_gate.enabled`` 构造 ``QualityGateNode``."""
    if not runner.config.quality_gate.enabled:
        return None
    zoo_path = (
        Path(runner.config.quality_gate.zoo_path)
        if runner.config.quality_gate.zoo_path
        else None
    )
    zoo = FactorZoo(zoo_path) if zoo_path is not None else FactorZoo()
    return QualityGateNode(QualityGateSetting(), zoo=zoo)


def build_trajectory_pool(runner: "PipelineRunner") -> Optional[TrajectoryPool]:
    """根据 ``cfg.evolution.enabled`` 构造 ``TrajectoryPool``."""
    if not runner.config.evolution.enabled:
        return None
    if runner.config.evolution.pool_dir:
        base = Path(runner.config.evolution.pool_dir)
    else:
        base = Path(runner.config.output.dir) / "trajectory"
    return TrajectoryPool(base)


def build_evolution_loop(
    runner: "PipelineRunner",
    pool: TrajectoryPool,
    quality_gate: Optional[QualityGateNode],
    workers: int = 1,
) -> EvolutionLoop:
    """构造 ``EvolutionLoop``, evaluate_fn 委托给 ``runner._evaluate_candidate``."""
    settings = EvolutionSetting(
        enabled=True,
        max_rounds=runner.config.evolution.max_rounds,
        parents_per_round=runner.config.evolution.parents_per_round,
        parent_selection_strategy=runner.config.evolution.parent_selection_strategy,
        top_percent_threshold=runner.config.evolution.top_percent_threshold,
        metric=runner.config.evolution.metric,
        early_stop_patience=runner.config.evolution.early_stop_patience,
    )
    return EvolutionLoop(
        settings=settings,
        pool=pool,
        quality_gate=quality_gate,
        evaluate_fn=runner._evaluate_candidate,
        workers=workers,
    )


def evaluate_candidate(
    runner: "PipelineRunner",
    candidate: FactorCandidate,
) -> tuple[bool, dict, FactorFeedback]:
    """``EvolutionLoop`` 评估回调: 单次回测 + metrics + feedback.

    Returns:
        ``(passed, metrics, feedback)``
    """
    if not isinstance(candidate, FactorCandidate):
        raise TypeError(f"expected FactorCandidate, got {type(candidate)}")

    try:
        ctx = run_one_factor(runner, candidate)
    except Exception as e:  # noqa: BLE001
        return False, {}, FactorFeedback(
            factor_id=candidate.factor_id,
            factor_name=candidate.name,
            decision=False,
            summary=f"evaluate failed: {e}",
        )

    passed = bool(ctx.get("status") != "rejected")
    metrics = extract_metrics_from_ctx(ctx)
    factor_id = str(candidate.factor_id)
    factor_name = str(candidate.name)
    feedback = FactorFeedback(
        factor_id=factor_id,
        factor_name=factor_name,
        decision=passed,
        summary="ok" if passed else "rejected",
        metadata=metrics,
    )
    if runner.config.feedback.enabled and "Feedback" in ctx:
        for node_fb in ctx["Feedback"].values():
            for ch, ch_fb in node_fb.channels.items():
                feedback.channels[ch] = ch_fb
    return passed, metrics, feedback


def run_one_factor(runner: "PipelineRunner", candidate: FactorCandidate) -> dict:
    """执行单次回测 (12 节点), 临时把 candidate 的 expression / name 注入 ``cfg.factor``.

    不写 TrajectoryPool, 仅返回 ctx.
    """
    if not isinstance(candidate, FactorCandidate):
        raise TypeError(f"expected FactorCandidate, got {type(candidate)}")
    original_name = runner.config.factor.name
    runner.config.factor.name = candidate.name
    if not getattr(runner.config.factor, "expression", ""):
        try:
            runner.config.factor.expression = candidate.expression  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        return runner.run()
    finally:
        runner.config.factor.name = original_name


def run_evolution(
    runner: "PipelineRunner",
    initial_directions: list[str] | None = None,
    initial_candidates: list[FactorCandidate] | None = None,
    workers: int = 1,
) -> EvolutionResult:
    """多轮演化主入口.

    Args:
        runner: ``PipelineRunner`` 实例
        initial_directions: round 0 用的研究假设列表 (Hypothesizer 处理)
        initial_candidates: round 0 用的直接候选 (跳过 Hypothesizer)
        workers: 并行数 (1=串行, >1=ThreadPool/ProcessPool 并行)

    Returns:
        ``EvolutionResult``: best entries + 统计

    Raises:
        ValueError: ``cfg.evolution.enabled=False``
    """
    if not runner.config.evolution.enabled:
        raise ValueError("config.evolution.enabled=False, 无法运行演化")

    pool = build_trajectory_pool(runner)
    quality_gate = build_quality_gate(runner)
    loop = build_evolution_loop(runner, pool, quality_gate, workers=workers)

    # workers > 1 + 有 _loader → 预序列化供 ProcessPool
    if workers > 1 and "_loader" in runner._context.get("LoadData", {}):
        snapshot = prepare_snapshot(
            runner.config, runner._context,
            factor_path=getattr(runner.config.factor, "factor_dir", None),
        )
        snap_path = Path(pool.base_dir) / "_snapshot.pkl"
        snapshot.save(snap_path)
        loop.snapshot_path = str(snap_path)
        logger.info("  [ProcessPool] 预序列化快照: %s", snap_path)

    # 连接 MetricCollector (streaming)
    collector = MetricCollector()
    loop.metric_collector = collector

    result = loop.run(
        initial_directions=initial_directions,
        initial_candidates=initial_candidates,
    )

    # 演化结束后, 追加写入 JSON + 生成 streaming dashboard
    if pool.size > 0:
        metrics_json = pool.base_dir / "metrics.json"
        collector.append_json(metrics_json)
        logger.info("  [Streaming] 指标追加: %s", metrics_json)
        dashboard_html = pool.base_dir.parent / "dashboard_streaming.html"
        generate_dashboard_html(
            collector,
            title=f"演化 Dashboard (Streaming): {runner.config.factor.name}",
            output_path=str(dashboard_html),
            streaming=True,
        )
        logger.info("  [Streaming] Dashboard: %s", dashboard_html)

    return result
