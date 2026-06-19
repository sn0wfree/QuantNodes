# coding: utf-8
"""Pipeline Runner / 管线编排器

混合模式数据传递:
- Phase 1 (严格串联): LoadData >> SampleFilter >> TradabilityFilter >> AdjustDate >> Preprocess >> Neutralize
- Phase 2 (Context 共享): ICAnalyzer / GroupAnalyzer / FactorScore / RiskCorrelation
- Phase 3 (依赖分析): LongShort
- Phase 4 (输出): FactorTestReport

FactorFeedback 集成 (Week 1.5, 可选):
- feedback.enabled=False: 现有行为完全不变 (向后兼容)
- feedback.enabled=True:  5 个分析节点返回值自动包装为 FactorFeedback,
  聚合到 ctx['Feedback'], 可选持久化到 feedback.output_dir

Phase R2 重构 (2026-06-19):
- 12 阶段从手写 137 行缩减为 PIPELINE_SPEC 数据驱动 (~30 行 run loop)
- Feedback 包装拆到 feedback_wrapper.py
- Evolution 适配拆到 evolution_adapter.py
- 指标提取拆到 utils/metrics_extractor.py
- 公共 API (`PipelineRunner.run()` / `.run_evolution()`) 完全兼容
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.research.factor_test.config import SingleFactorTestConfig
from QuantNodes.research.factor_test.evolution_adapter import (
    build_evolution_loop,
    build_quality_gate,
    build_trajectory_pool,
    evaluate_candidate,
    run_evolution as _run_evolution,
    run_one_factor,
)
from QuantNodes.research.factor_test.feedback_wrapper import (
    ANALYSIS_NODES as _ANALYSIS_NODES,  # noqa: F401  re-export 兼容
    build_feedback,
    maybe_build_judge,
    maybe_persist_feedback,
)
from QuantNodes.research.factor_test.pipeline_spec import PIPELINE_SPEC, PhaseSpec
from QuantNodes.research.factor_test.utils.metrics_extractor import (
    extract_metrics_from_ctx as _extract_metrics_from_ctx,  # noqa: F401  re-export
)

if TYPE_CHECKING:
    from QuantNodes.core.evolution import EvolutionResult, FactorCandidate
    from QuantNodes.core.feedback import LLMJudge
    from QuantNodes.core.quality_gate import QualityGateNode
    from QuantNodes.core.trajectory import TrajectoryPool


class PipelineRunner:
    """单因子回测管线编排器

    用法:
        config = SingleFactorTestConfig(...)
        runner = PipelineRunner(config)
        result = runner.run()

        # 或从 YAML:
        runner = PipelineRunner.from_yaml("config.yaml")
        result = runner.run()

    FactorFeedback 集成:
        当 config.feedback.enabled=True 时, 5 个分析节点返回值会自动包装为
        FactorFeedback, 聚合到 ctx['Feedback'] = {node_name: FactorFeedback}。
        若 config.feedback.output_dir 不为 None, 还会持久化到该目录。
    """

    def __init__(self, config: SingleFactorTestConfig):
        self.config = config
        self._context: dict = {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineRunner":
        """从 YAML 配置文件创建 Runner"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(SingleFactorTestConfig(**raw))

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineRunner":
        """从 dict 创建 Runner"""
        return cls(SingleFactorTestConfig(**data))

    # ============================================================
    # Phase R2: 声明式 12 节点 run loop
    # ============================================================

    def run(self) -> dict:
        """执行完整单因子回测管线

        Returns:
            dict: 完整结果, 各节点输出按 ``ctx[node_name]`` 索引
                  当 ``config.feedback.enabled=True`` 时, 还包含 'Feedback' 键
        """
        cfg = self.config
        ctx, pre_seeded = self._seed_ctx()

        feedback_enabled = cfg.feedback.enabled
        factor_id = str(uuid.uuid4())
        factor_name = cfg.factor.name
        judge = maybe_build_judge(cfg) if feedback_enabled else None

        print("=" * 60)
        print(f"单因子回测: {cfg.factor.name}")
        print(f"时间范围: {cfg.preprocess.adj_date_beg} ~ {cfg.preprocess.adj_date_end}")
        if feedback_enabled:
            print(f"FactorFeedback: ENABLED (factor_id={factor_id[:8]}...)")
        print("=" * 60)

        for spec in PIPELINE_SPEC:
            self._run_phase(spec, ctx, pre_seeded)

        # FactorFeedback 自动包装 (可选)
        if feedback_enabled:
            ctx["Feedback"] = build_feedback(ctx, factor_id, factor_name, cfg, judge)
            maybe_persist_feedback(ctx["Feedback"], cfg)
            n_passed = sum(1 for fb in ctx["Feedback"].values() if fb.decision)
            print(f"\n[Feedback] 包装完成: {len(ctx['Feedback'])} 节点, {n_passed} 通过")

        print("\n" + "=" * 60)
        print("单因子回测完成!")
        print("=" * 60)
        return ctx

    def _seed_ctx(self) -> tuple[dict, bool]:
        """初始化 ctx; 若已注入 LoadData, 复用之, 否则空 dict.

        Returns:
            (ctx, pre_seeded) — pre_seeded=True 表示 LoadData 已外部注入
        """
        if "LoadData" in self._context:
            ctx = dict(self._context)
            shape = ctx["LoadData"].get("factor", pd.DataFrame()).shape
            print(f"  [LoadData 跳过] 使用已注入数据 (factor shape: {shape})")
            return ctx, True
        return {}, False

    def _run_phase(self, spec: PhaseSpec, ctx: dict, pre_seeded: bool) -> None:
        """执行单个阶段 (含 skip / log)."""
        if spec.skip_if_in_ctx and spec.name in ctx:
            return
        print(f"\n[Phase {spec.phase_no}] {spec.title}...")
        node = spec.node_cls(config=spec.build_cfg(self.config))
        ctx[spec.name] = node.execute(context=ctx)
        if spec.log_summary is not None:
            line = spec.log_summary(self.config, ctx[spec.name])
            if line:
                print(line)

    # ============================================================
    # 兼容旧 API: 私有方法 facade re-export
    # ============================================================

    def _build_feedback(
        self,
        ctx: dict,
        factor_id: str,
        factor_name: str,
        judge: Optional["LLMJudge"],
    ) -> dict:
        """向后兼容: 委托给 ``feedback_wrapper.build_feedback``."""
        return build_feedback(ctx, factor_id, factor_name, self.config, judge)

    def _maybe_persist_feedback(
        self,
        feedbacks: dict,
        cfg: SingleFactorTestConfig,
    ) -> None:
        """向后兼容: 委托给 ``feedback_wrapper.maybe_persist_feedback``."""
        maybe_persist_feedback(feedbacks, cfg)

    @staticmethod
    def _maybe_build_judge(cfg: SingleFactorTestConfig) -> Optional["LLMJudge"]:
        """向后兼容: 委托给 ``feedback_wrapper.maybe_build_judge``."""
        return maybe_build_judge(cfg)

    # ============================================================
    # Week 4: 演化集成 (向后兼容 facade, 实现在 evolution_adapter.py)
    # ============================================================

    def _build_quality_gate(self) -> Optional["QualityGateNode"]:
        return build_quality_gate(self)

    def _build_trajectory_pool(self) -> Optional["TrajectoryPool"]:
        return build_trajectory_pool(self)

    def _build_evolution_loop(
        self,
        pool: "TrajectoryPool",
        quality_gate: Optional["QualityGateNode"],
        workers: int = 1,
    ):
        return build_evolution_loop(self, pool, quality_gate, workers=workers)

    def _evaluate_candidate(
        self,
        candidate: "FactorCandidate",
    ) -> tuple[bool, dict, "FactorFeedback"]:
        return evaluate_candidate(self, candidate)

    def _run_one_factor(self, candidate: "FactorCandidate") -> dict:
        return run_one_factor(self, candidate)

    def run_evolution(
        self,
        initial_directions: list[str] | None = None,
        initial_candidates: list["FactorCandidate"] | None = None,
        workers: int = 1,
    ) -> "EvolutionResult":
        """多轮演化主入口 (委托给 ``evolution_adapter.run_evolution``)."""
        return _run_evolution(
            self,
            initial_directions=initial_directions,
            initial_candidates=initial_candidates,
            workers=workers,
        )

    @property
    def context(self) -> dict:
        """获取当前上下文"""
        return self._context


# ============================================================
# 便捷函数
# ============================================================


def run_single_factor_test(config: dict) -> dict:
    """便捷函数: 从 dict 配置运行单因子回测

    Args:
        config: 配置字典

    Returns:
        dict: 完整结果
    """
    runner = PipelineRunner.from_dict(config)
    return runner.run()


def run_single_factor_test_yaml(yaml_path: str) -> dict:
    """便捷函数: 从 YAML 配置运行单因子回测

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        dict: 完整结果
    """
    runner = PipelineRunner.from_yaml(yaml_path)
    return runner.run()
