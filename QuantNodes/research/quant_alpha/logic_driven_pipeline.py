# coding=utf-8
"""
logic_driven_pipeline.py - 逻辑驱动的端到端 Pipeline

基于 AlphaLogics 论文 (arXiv 2603.20247) §3.2/§3.3 + PR-5 端到端集成。

Pipeline 流程:
  AlphaLogicsWorkflow (外层循环 + 内层循环)
    ↓
  输出: 最佳逻辑 H* + 该逻辑下的因子池
    ↓
  MCTS (可选, 基于最佳逻辑的 Gamma 约束继续优化)
    ↓
  Dedup (Mutual IC)
    ↓
  Wiki 持久化

Usage::

    from QuantNodes.research.quant_alpha.logic_driven_pipeline import (
        LogicDrivenPipeline, LogicDrivenPipelineConfig,
    )

    config = LogicDrivenPipelineConfig(
        objective="capture A-share reversal effect",
        max_outer_rounds=3,
    )
    pipeline = LogicDrivenPipeline(config=config)
    result = pipeline.run(data)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.research.quant_alpha.logic_mining import (
    WikiLogicStructured,
    LogicPerformanceEvidence,
)
from QuantNodes.research.quant_alpha.pipeline import (
    AlphaPipeline,
    PipelineConfig,
    PipelineResult,
    TerminationConfig,
    RoundResult,
)
from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
    AlphaLogicsConfig,
    AlphaLogicsWorkflow,
    AlphaLogicsResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LogicDrivenPipelineConfig",
    "LogicDrivenPipelineResult",
    "LogicDrivenPipeline",
]


@dataclass
class LogicDrivenPipelineConfig:
    """逻辑驱动 Pipeline 配置

    当 logic_driven=True 时,使用 AlphaLogicsWorkflow (外层+内层循环);
    否则,使用原 AlphaPipeline (单轮内层循环)。
    """
    # 基础配置 (与 PipelineConfig 类似)
    objective: str = ""
    wiki_path: str = "wiki/"
    output_dir: str = "pipeline_output_logic_driven"

    # 逻辑驱动开关
    logic_driven: bool = True

    # Alpha-GPT 配置
    alphagpt_iterations: int = 3
    alphagpt_pool_size: int = 10
    alphagpt_top_k: int = 10

    # MCTS 配置
    mcts_iterations: int = 50
    mcts_max_depth: int = 5
    mcts_dedup_threshold: float = 0.7

    # 去重
    max_mutual_ic: float = 0.7
    min_ir_threshold: float = 0.5

    # 评估
    min_ic_decay_ratio: float = 0.3
    max_turnover: float = 2.0

    # 通用
    top_k: int = 10
    date_column: str = "date"
    code_column: str = "code"
    forward_returns: Tuple[int, ...] = (1, 5, 20)

    # LLM
    llm_provider: str = "minimax"
    llm_model: Optional[str] = None
    temperature: float = 0.7

    # 各阶段温度
    temperature_idea_gen: float = 0.8
    temperature_formula: float = 0.4
    temperature_reflector: float = 0.6
    temperature_critic: float = 0.3

    # 多轮迭代
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    # AlphaLogics 配置
    alphalogics_inner_iterations: int = 2
    alphalogics_inner_pool_size: int = 5
    alphalogics_max_outer_rounds: int = 3
    alphalogics_initial_sources: Tuple[str, ...] = ("alpha101", "alpha158")
    alphalogics_initial_max_per_lib: int = 3

    # 终止条件
    timeout_seconds: int = 3600

    # 一致性评分
    consistency_use_llm: bool = False  # 是否使用真实 LLM 评分（False=结构化匹配）
    consistency_score_threshold: float = 0.5


@dataclass
class LogicDrivenPipelineResult:
    """逻辑驱动 Pipeline 结果"""
    # AlphaLogics 结果
    alphalogics_result: Optional[AlphaLogicsResult] = None

    # 最佳逻辑及其 Γ 约束
    best_logic_name: Optional[str] = None
    best_logic_structured: Optional[WikiLogicStructured] = None
    best_evidence: Optional[LogicPerformanceEvidence] = None
    best_gamma: Optional[Any] = None  # CompiledConstraint

    # 因子池
    logic_driven_factors: List[Any] = field(default_factory=list)  # FactorMetrics

    # MCTS 优化结果 (可选)
    mcts_enhanced: bool = False
    final_pool: List[Any] = field(default_factory=list)

    # Wiki
    wiki_pages: List[str] = field(default_factory=list)

    # 元数据
    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


class LogicDrivenPipeline:
    """逻辑驱动的端到端 Pipeline

    整合 AlphaLogics 外层循环 + MCTS 优化 + Wiki 持久化。
    """

    def __init__(
        self,
        config: LogicDrivenPipelineConfig,
        llm_client: Any = None,
    ):
        self.config = config
        self.llm_client = llm_client

    def run(self, data: pl.DataFrame) -> LogicDrivenPipelineResult:
        """运行 Pipeline

        Args:
            data: 行情数据

        Returns:
            LogicDrivenPipelineResult
        """
        start = time.time()
        result = LogicDrivenPipelineResult()

        if not self.config.logic_driven:
            logger.info("logic_driven=False, fall back to standard AlphaPipeline")
            return self._run_standard_pipeline(data, result, start)

        # 逻辑驱动模式
        logger.info("=== 逻辑驱动 Pipeline ===")

        # Step 1: AlphaLogics 外层循环
        alphalogics_result = self._run_alphalogics(data)
        result.alphalogics_result = alphalogics_result
        result.best_logic_name = (
            alphalogics_result.best_logic.name
            if alphalogics_result.best_logic else None
        )
        result.best_evidence = alphalogics_result.best_evidence

        if alphalogics_result.best_logic is None:
            logger.warning("AlphaLogics returned no best logic, terminating")
            result.elapsed_seconds = time.time() - start
            result.summary = {"error": "no_best_logic"}
            return result

        # Step 2: 获取最佳逻辑的结构化字段
        best_logic = alphalogics_result.best_logic
        result.best_logic_structured = best_logic.structured

        # Step 3: 编译 Γ 约束
        if best_logic.structured is None:
            logger.warning("Best logic has no structured fields, skipping MCTS")
            # 直接使用内层因子池
            for ir in alphalogics_result.inner_results:
                if ir.alphagpt_result:
                    from QuantNodes.research.quant_alpha.evaluation.contracts import (
                        FactorMetrics,
                    )
                    for f in ir.alphagpt_result.final_pool:
                        result.logic_driven_factors.append(FactorMetrics(
                            formula_id=f.formula_id,
                            status="success",
                            ic_mean=f.ic_mean,
                            ir=f.ir,
                            overall_score=f.ir,
                        ))
        else:
            from QuantNodes.research.quant_alpha.logic_mining import compile_to_constraint
            gamma = compile_to_constraint(
                best_logic.structured,
                source_logic=best_logic.name,
            )
            result.best_gamma = gamma

            # Step 4: 使用最佳 Γ 约束运行 MCTS 优化
            self._enhance_with_mcts(data, gamma, result)

        # Step 5: Wiki 持久化
        if result.final_pool:
            self._persist_to_wiki(result)

        # 汇总
        result.elapsed_seconds = time.time() - start
        result.summary = self._build_summary(result)
        logger.info(
            "=== 逻辑驱动 Pipeline 完成: best_logic=%s, factors=%d, %.1fs ===",
            result.best_logic_name,
            len(result.final_pool),
            result.elapsed_seconds,
        )
        return result

    def _run_alphalogics(self, data: pl.DataFrame) -> AlphaLogicsResult:
        """运行 AlphaLogics 外层循环"""
        config = AlphaLogicsConfig(
            inner_iterations=self.config.alphalogics_inner_iterations,
            inner_pool_size=self.config.alphalogics_inner_pool_size,
            inner_early_stop=self.config.termination.patience,
            max_outer_rounds=self.config.alphalogics_max_outer_rounds,
            inner_objective="ir",
            data=data,
            date_column=self.config.date_column,
            code_column=self.config.code_column,
            forward_returns=self.config.forward_returns,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            wiki_path=self.config.wiki_path,
            persist_best_logic=True,
            initial_logic_sources=self.config.alphalogics_initial_sources,
            initial_logic_max_per_lib=self.config.alphalogics_initial_max_per_lib,
            min_ir_threshold=self.config.min_ir_threshold,
        )
        workflow = AlphaLogicsWorkflow(config=config, llm_client=self.llm_client)
        return workflow.run()

    def _enhance_with_mcts(
        self,
        data: pl.DataFrame,
        gamma: Any,
        result: LogicDrivenPipelineResult,
    ) -> None:
        """使用最佳 Γ 约束增强 MCTS"""
        # 收集内层因子
        for ir in result.alphalogics_result.inner_results:
            if ir.alphagpt_result:
                from QuantNodes.research.quant_alpha.evaluation.contracts import (
                    FactorMetrics,
                )
                for f in ir.alphagpt_result.final_pool:
                    result.logic_driven_factors.append(FactorMetrics(
                        formula_id=f.formula_id,
                        status="success",
                        ic_mean=f.ic_mean,
                        ir=f.ir,
                        overall_score=f.ir,
                    ))

        # 使用 Gamma 约束运行 Pipeline (单轮 + MCTS)
        pipeline_config = PipelineConfig(
            objective=self.config.objective,
            wiki_path=self.config.wiki_path,
            termination=self.config.termination,
            alphagpt_iterations=self.config.alphagpt_iterations,
            alphagpt_pool_size=self.config.alphagpt_pool_size,
            mcts_iterations=self.config.mcts_iterations,
            mcts_max_depth=self.config.mcts_max_depth,
            max_mutual_ic=self.config.max_mutual_ic,
            min_ir_threshold=self.config.min_ir_threshold,
            top_k=self.config.top_k,
            date_column=self.config.date_column,
            code_column=self.config.code_column,
            forward_returns=self.config.forward_returns,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            gamma=gamma,
            output_dir=self.config.output_dir,
        )
        pipeline = AlphaPipeline(pipeline_config)
        pipeline_result = pipeline.run(data)

        result.final_pool = pipeline_result.final_pool
        result.mcts_enhanced = True

    def _run_standard_pipeline(
        self,
        data: pl.DataFrame,
        result: LogicDrivenPipelineResult,
        start: float,
    ) -> LogicDrivenPipelineResult:
        """回退到标准 Pipeline"""
        pipeline_config = PipelineConfig(
            objective=self.config.objective,
            wiki_path=self.config.wiki_path,
            termination=self.config.termination,
            alphagpt_iterations=self.config.alphagpt_iterations,
            alphagpt_pool_size=self.config.alphagpt_pool_size,
            mcts_iterations=self.config.mcts_iterations,
            mcts_max_depth=self.config.mcts_max_depth,
            max_mutual_ic=self.config.max_mutual_ic,
            min_ir_threshold=self.config.min_ir_threshold,
            top_k=self.config.top_k,
            date_column=self.config.date_column,
            code_column=self.config.code_column,
            forward_returns=self.config.forward_returns,
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
            output_dir=self.config.output_dir,
        )
        pipeline = AlphaPipeline(pipeline_config)
        pipeline_result = pipeline.run(data)

        result.final_pool = pipeline_result.final_pool
        result.elapsed_seconds = time.time() - start
        result.summary = pipeline_result.summary
        return result

    def _persist_to_wiki(self, result: LogicDrivenPipelineResult) -> None:
        """持久化到 Wiki"""
        try:
            from QuantNodes.research.wiki.enums import (
                FactorCategory,
                FactorSource,
            )
            from QuantNodes.research.wiki.factor import WikiFactor
            from QuantNodes.research.wiki.proxy import WikiFactorProxy
            wiki = WikiFactorProxy(self.config.wiki_path)
            for f in result.final_pool:
                try:
                    wiki_factor = WikiFactor(
                        name=f.formula_id,
                        formula=getattr(f, "formula", f.formula_id),
                        source=FactorSource.AUTO_RESEARCH,
                        category=FactorCategory.OTHER,
                        tags=[
                            "logic-driven",
                            f"ir={f.ir:.3f}",
                            f"logic={result.best_logic_name}",
                        ],
                        # WikiFactor V2: 主动填充新字段 (logic-driven 出因子 = draft, 等后续验证)
                        factor_params={"logic": result.best_logic_name, "source_pipeline": "logic-driven"},
                        status="draft",
                        ic_mean=f.ic_mean,
                        ic_std=f.ic_std,
                        icir=f.ir,
                        rank_ic_mean=f.rank_ic_mean,
                        description=f"Logic-driven factor from {result.best_logic_name}",
                    )
                    page_name = wiki.store_factor(wiki_factor)
                    result.wiki_pages.append(page_name)
                except Exception as e:
                    logger.warning("Wiki store_factor failed for %s: %s", f.formula_id, e)
        except Exception as e:
            logger.warning("Wiki persistence failed: %s", e)

    def _build_summary(self, result: LogicDrivenPipelineResult) -> Dict[str, Any]:
        """构建摘要"""
        return {
            "logic_driven": True,
            "best_logic_name": result.best_logic_name,
            "best_ir": result.best_evidence.best_ir if result.best_evidence else 0.0,
            "best_n_factors": (
                result.best_evidence.n_factors_explored
                if result.best_evidence else 0
            ),
            "logic_driven_factors": len(result.logic_driven_factors),
            "mcts_enhanced": result.mcts_enhanced,
            "final_factors": len(result.final_pool),
            "wiki_pages": len(result.wiki_pages),
            "elapsed_seconds": result.elapsed_seconds,
        }