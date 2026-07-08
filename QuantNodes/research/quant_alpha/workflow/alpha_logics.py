# coding=utf-8
"""
alpha_logics.py - 外层循环编排

基于 AlphaLogics 论文 (arXiv 2603.20247) §3.3 实现外层循环：
- 内层：Alpha-GPT 在固定逻辑 H 下生成因子（受 Γ 约束）
- 外层：聚合 per-logic 证据 → 重构/新增逻辑 → 跨轮持久化

外层循环（对齐论文 Algorithm 2）:

  Input: ℋ_init, T (max outer rounds)
  Output: H* (最优逻辑)

  ℋ_lib = ℋ_init
  H_current = generator(ℋ_lib)
  H_best = H_current; E_best = None
  for t in range(T):
    (E^Logic, R*) = inner_loop(H_current)        # 内层
    fb_logic = refinement_agent(H_current, ℰ_hist)  # 反馈
    if E_best is None or R* > E_best:
      H_best = H_current; E_best = E^Logic
      wiki.update_logic_evidence(H_best.name, E_best)
    H_new = generator(ℋ_lib, H_current, ℰ_hist)   # 生成下一条
    ℋ_lib.append(H_new)
    H_current = H_new
  return H*

Usage::

    from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
        AlphaLogicsConfig, AlphaLogicsWorkflow, AlphaLogicsResult,
    )

    config = AlphaLogicsConfig(
        inner_iterations=5,
        max_outer_rounds=4,
        initial_logic_sources=("alpha101", "alpha158"),
    )
    workflow = AlphaLogicsWorkflow(config=config)
    result = workflow.run()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.research.quant_alpha.logic_mining import (
    LogicMiningPipeline,
    WikiLogicStructured,
    LogicPerformanceEvidence,
    LogicAbstractionResult,
    build_initial_logic_library,
)
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
    MarketLogicRefinementDirection,
    generate_logic_name,
)
from QuantNodes.research.quant_alpha.workflow import (
    AlphaGptConfig,
    AlphaGptWorkflow,
    AlphaGptResult,
)

# 延迟导入避免循环
def _get_wiki_proxy():
    from QuantNodes.research.wiki.proxy import WikiFactorProxy
    return WikiFactorProxy

def _get_wiki_logic():
    from QuantNodes.research.wiki.logic import WikiLogic
    return WikiLogic

def _get_logic_source():
    from QuantNodes.research.wiki.enums import LogicSource
    return LogicSource

logger = logging.getLogger(__name__)

__all__ = [
    "AlphaLogicsConfig",
    "AlphaLogicsWorkflow",
    "AlphaLogicsResult",
    "InnerLoopResult",
]


@dataclass
class AlphaLogicsConfig:
    """AlphaLogicsWorkflow 配置

    内层（Alpha-GPT）+ 外层（逻辑重构）参数。
    """
    # 内层参数
    inner_iterations: int = 5
    inner_pool_size: int = 10
    inner_early_stop: int = 3
    inner_objective: str = "ir"  # "ir" / "ic" / "icir"

    # 外层参数
    max_outer_rounds: int = 4

    # 数据
    data: Optional[pl.DataFrame] = None
    data_path: Optional[str] = None
    date_column: str = "date"
    code_column: str = "code"
    forward_returns: Tuple[int, ...] = (1, 5, 20)

    # LLM
    llm_provider: str = "minimax"
    llm_model: Optional[str] = None

    # 持久化
    wiki_path: str = "wiki"
    persist_best_logic: bool = True

    # 初始逻辑库
    initial_logic_sources: Tuple[str, ...] = ("alpha101", "alpha158")
    initial_logic_max_per_lib: int = 5

    # 评估阈值
    min_ir_threshold: float = 0.1


@dataclass
class InnerLoopResult:
    """内层循环结果"""
    logic_name: str
    alphagpt_result: Optional[AlphaGptResult] = None
    evidence: Optional[LogicPerformanceEvidence] = None
    elapsed_seconds: float = 0.0


@dataclass
class AlphaLogicsResult:
    """外层循环最终结果"""
    best_logic: Optional[WikiLogic] = None
    best_evidence: Optional[LogicPerformanceEvidence] = None
    library: List[WikiLogic] = field(default_factory=list)
    history: List[WikiLogic] = field(default_factory=list)
    evidence_history: List[LogicPerformanceEvidence] = field(default_factory=list)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    inner_results: List[InnerLoopResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


def _build_inner_evidence(
    logic_name: str,
    alphagpt_result: AlphaGptResult,
    round_idx: int,
) -> LogicPerformanceEvidence:
    """从内层 Alpha-GPT 结果构建 per-logic 证据"""
    evaluations = alphagpt_result.summary if alphagpt_result else {}
    irs = [
        f.ir for f in (alphagpt_result.final_pool if alphagpt_result else [])
        if hasattr(f, "ir")
    ]
    if not irs:
        return LogicPerformanceEvidence(
            n_factors_explored=0,
            best_ir=0.0,
            best_ic=0.0,
            mean_ir=0.0,
            refinement_round=round_idx,
        )

    best_ir = max(irs)
    best_idx = irs.index(best_ir)
    best_factor_id = (
        alphagpt_result.final_pool[best_idx].formula_id
        if alphagpt_result and alphagpt_result.final_pool
        else None
    )

    return LogicPerformanceEvidence(
        n_factors_explored=len(alphagpt_result.final_pool) if alphagpt_result else 0,
        best_ir=float(best_ir),
        best_ic=float(best_ir),  # 简化：暂用 IR 作为 IC proxy
        best_factor_id=best_factor_id,
        mean_ir=float(sum(irs) / len(irs)) if irs else 0.0,
        refinement_round=round_idx,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


class AlphaLogicsWorkflow:
    """AlphaLogics 外层循环 Workflow

    编排论文 Algorithm 2：内层循环 → 证据聚合 → 重构 → 跨轮持久化。
    """

    def __init__(
        self,
        config: AlphaLogicsConfig,
        llm_client: Any = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.wiki = _get_wiki_proxy()(config.wiki_path)
        self.mining_pipeline = LogicMiningPipeline(llm_client=llm_client)
        self.generator = MarketLogicGenerator(
            llm_client=llm_client, base_name="alpha_logic"
        )
        self.refiner = MarketLogicRefinementDirection(llm_client=llm_client)

    def run(self) -> AlphaLogicsResult:
        """运行外层循环

        Returns:
            AlphaLogicsResult
        """
        start = time.time()
        result = AlphaLogicsResult()

        # Step 1: 构建初始逻辑库
        logger.info("=== AlphaLogics 外层循环 ===")
        logger.info("Step 1: 构建初始逻辑库")
        library = self._build_initial_library()
        result.library = library.copy()
        logger.info("初始库: %d logics", len(library))

        if not library:
            logger.warning("初始库为空，终止")
            result.elapsed_seconds = time.time() - start
            result.summary = {"error": "initial_library_empty"}
            return result

        # Step 2: 外层循环
        h_current = library[0]
        h_hist = []
        e_hist = []
        fb_hist = []
        h_best = h_current
        e_best = None

        for t in range(1, self.config.max_outer_rounds + 1):
            logger.info("=== Round %d/%d: %s ===", t, self.config.max_outer_rounds, h_current.name)

            # === 内层循环 ===
            inner_result = self._run_inner_loop(h_current, t)
            result.inner_results.append(inner_result)
            e_logic = inner_result.evidence
            e_hist.append(e_logic)
            logger.info(
                "  Inner: best_ir=%.4f, n_factors=%d",
                e_logic.best_ir if e_logic else 0.0,
                e_logic.n_factors_explored if e_logic else 0,
            )

            # === 反馈 ===
            fb = self.refiner.refine(h_current, h_hist, e_hist)
            fb_hist.append(fb)
            logger.info("  Feedback: %s / %s", fb["diagnosis"], fb["direction"])

            # === 选最优 ===
            if e_best is None or (e_logic and e_logic.best_ir > e_best.best_ir):
                h_best = h_current
                e_best = e_logic

                # 持久化到 Wiki
                if self.config.persist_best_logic and e_logic:
                    h_best.performance_evidence = e_logic
                    try:
                        self.wiki.store_logic(h_best)
                    except Exception as ex:
                        logger.warning("Wiki store_logic failed: %s", ex)
                logger.info("  New best: %s (ir=%.4f)", h_best.name, e_best.best_ir)

            # === 生成下一条 ===
            h_hist.append(h_current)
            h_new = self.generator.generate(
                library=library,
                current_logic=h_current,
                history=h_hist,
                evidence=e_hist,
                round_idx=t + 1,
            )

            # 持久化新逻辑
            if self.config.persist_best_logic:
                try:
                    self.wiki.store_logic(h_new)
                except Exception as ex:
                    logger.warning("Wiki store_logic failed: %s", ex)

            library.append(h_new)
            result.library.append(h_new)
            h_current = h_new

        # 汇总
        result.best_logic = h_best
        result.best_evidence = e_best
        result.history = h_hist
        result.evidence_history = e_hist
        result.feedback_history = fb_hist
        result.elapsed_seconds = time.time() - start
        result.summary = self._build_summary(result)
        logger.info(
            "=== AlphaLogics 完成: best=%s, ir=%.4f, %.1fs ===",
            h_best.name if h_best else "None",
            e_best.best_ir if e_best else 0.0,
            result.elapsed_seconds,
        )
        return result

    def _build_initial_library(self) -> List[WikiLogic]:
        """构建初始逻辑库"""
        logics = build_initial_logic_library(
            source_libs=self.config.initial_logic_sources,
            llm_client=self.llm_client,
            max_per_lib=self.config.initial_logic_max_per_lib,
            only_volume_price=True,
        )

        wiki_logics = []
        for i, result in enumerate(logics):
            if result.structured_logic is None:
                continue
            name = generate_logic_name(
                f"alpha_logic_{result.source_lib}", i + 1
            )
            WikiLogic = _get_wiki_logic()
            LogicSource = _get_logic_source()
            logic = WikiLogic(
                name=name,
                content=f"Auto-mined from {result.source_lib}",
                source=LogicSource.RESEARCH_REPORT,
                extracted_formula=result.source_formula,
                validation_status="pending",
                structured=result.structured_logic,
                parent_logic=None,
                refinement_round=0,
            )
            wiki_logics.append(logic)
            if self.config.persist_best_logic:
                try:
                    self.wiki.store_logic(logic)
                except Exception as ex:
                    logger.warning("Wiki store_logic failed: %s", ex)

        return wiki_logics

    def _run_inner_loop(
        self,
        logic: WikiLogic,
        round_idx: int,
    ) -> InnerLoopResult:
        """运行内层循环（Alpha-GPT 在固定逻辑下生成因子）"""
        start = time.time()
        if logic.structured is None:
            return InnerLoopResult(
                logic_name=logic.name,
                evidence=LogicPerformanceEvidence(refinement_round=round_idx),
                elapsed_seconds=time.time() - start,
            )

        # 编译 Γ 约束
        from QuantNodes.research.quant_alpha.logic_mining import compile_to_constraint
        gamma = compile_to_constraint(logic.structured, source_logic=logic.name)

        # 配置 Alpha-GPT
        config = AlphaGptConfig(
            objective=f"Apply logic {logic.name}: {logic.content}",
            iterations=self.config.inner_iterations,
            pool_size=self.config.inner_pool_size,
            top_k=self.config.inner_pool_size,
            min_ir_threshold=self.config.min_ir_threshold,
            max_mutual_ic_threshold=0.7,
            forward_returns=list(self.config.forward_returns),
            date_column=self.config.date_column,
            code_column=self.config.code_column,
            gamma=gamma,
        )

        try:
            workflow = AlphaGptWorkflow(
                config=config,
                data=self.config.data,
                data_path=self.config.data_path,
                llm_client=self.llm_client,
            )
            alphagpt_result = workflow.run()
        except Exception as e:
            logger.warning("Inner loop Alpha-GPT failed: %s", e)
            alphagpt_result = None

        # 构建证据
        evidence = _build_inner_evidence(logic.name, alphagpt_result, round_idx)

        return InnerLoopResult(
            logic_name=logic.name,
            alphagpt_result=alphagpt_result,
            evidence=evidence,
            elapsed_seconds=time.time() - start,
        )

    def _build_summary(self, result: AlphaLogicsResult) -> Dict[str, Any]:
        """构建摘要"""
        return {
            "max_outer_rounds": self.config.max_outer_rounds,
            "rounds_completed": len(result.inner_results),
            "library_size": len(result.library),
            "best_logic": result.best_logic.name if result.best_logic else None,
            "best_ir": result.best_evidence.best_ir if result.best_evidence else 0.0,
            "best_n_factors": (
                result.best_evidence.n_factors_explored if result.best_evidence else 0
            ),
            "elapsed_seconds": result.elapsed_seconds,
        }