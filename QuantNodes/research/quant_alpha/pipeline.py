# coding=utf-8
"""
pipeline.py - 端到端因子挖掘流水线

连接 Alpha-GPT → MCTS → 去重 → Wiki。

Usage::

    from QuantNodes.research.quant_alpha.pipeline import AlphaPipeline, PipelineConfig

    config = PipelineConfig(
        objective="capture A-share reversal effect",
        wiki_path="wiki/",
        alphagpt_iterations=3,
        mcts_iterations=50,
    )
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
    deduplicate_mutual_ic,
)
from QuantNodes.research.quant_alpha.workflow import (
    AlphaGptConfig,
    AlphaGptWorkflow,
    AlphaGptResult,
)
from QuantNodes.research.quant_alpha.mcts.search import (
    MCTSSearch,
    MCTSSearchConfig,
    MCTSSearchResult,
)
from QuantNodes.research.quant_alpha.mcts.cache import MCTSCache, MCTSCacheConfig
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
from QuantNodes.research.wiki import (
    WikiFactorProxy,
    WikiFactor,
    FactorSource,
    FactorCategory,
)

logger = logging.getLogger(__name__)

__all__ = ["AlphaPipeline", "PipelineConfig", "PipelineResult"]


# ==============================================================================
# 配置
# ==============================================================================


@dataclass
class PipelineConfig:
    """流水线配置"""

    # 研究目标
    objective: str

    # Wiki 配置
    wiki_path: str = "wiki/"

    # Alpha-GPT 配置
    alphagpt_iterations: int = 3
    alphagpt_pool_size: int = 10
    alphagpt_top_k: int = 10

    # MCTS 配置
    mcts_iterations: int = 50
    mcts_max_depth: int = 5
    mcts_dedup_threshold: float = 0.7

    # 去重配置
    max_mutual_ic: float = 0.7

    # Alpha-GPT 过滤配置
    min_ir_threshold: float = 0.1

    # 通用配置
    top_k: int = 10
    date_column: str = "date"
    code_column: str = "code"
    forward_returns: Tuple[int, ...] = (1, 5, 20)

    # LLM 配置
    llm_provider: str = "minimax"
    llm_model: Optional[str] = None
    temperature: float = 0.7

    # 各阶段温度参数
    temperature_idea_gen: float = 0.8   # 鼓励创新
    temperature_formula: float = 0.4    # 需要精确
    temperature_reflector: float = 0.6  # 平衡
    temperature_critic: float = 0.3     # 需要稳定


# ==============================================================================
# 结果
# ==============================================================================


@dataclass
class PipelineResult:
    """流水线结果"""

    alphagpt_result: Optional[AlphaGptResult] = None
    mcts_result: Optional[MCTSSearchResult] = None
    final_pool: List[FactorMetrics] = field(default_factory=list)
    wiki_pages: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 流水线
# ==============================================================================


class AlphaPipeline:
    """端到端因子挖掘流水线

    连接 Alpha-GPT → MCTS → 去重 → Wiki。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vocab = OperatorVocab.default()
        self.wiki = WikiFactorProxy(config.wiki_path)

    def run(self, data: pl.DataFrame) -> PipelineResult:
        """运行流水线

        Args:
            data: 行情数据 DataFrame

        Returns:
            PipelineResult
        """
        start_time = time.time()
        result = PipelineResult()

        # 预计算前瞻收益（避免重复计算）
        logger.info("[Pipeline] 预计算前瞻收益...")
        data = self._precompute_forward_returns(data)

        # Stage 1: Alpha-GPT
        logger.info("[Pipeline] Stage 1: Alpha-GPT")
        result.alphagpt_result = self._run_alphagpt(data)

        # Stage 2: MCTS (使用 Alpha-GPT 结果作为种子)
        logger.info("[Pipeline] Stage 2: MCTS")
        seed_formulas = self._extract_seed_formulas(result.alphagpt_result)
        result.mcts_result = self._run_mcts(data, seed_formulas)

        # Stage 3: 合并去重
        logger.info("[Pipeline] Stage 3: 合并去重")
        result.final_pool = self._merge_and_dedup(
            result.alphagpt_result, result.mcts_result, data
        )

        # Stage 4: Wiki 持久化
        logger.info("[Pipeline] Stage 4: Wiki 持久化")
        result.wiki_pages = self._persist_to_wiki(result.final_pool)

        # 构建摘要
        result.elapsed_seconds = time.time() - start_time
        result.summary = self._build_summary(result)

        logger.info(
            "[Pipeline] 完成: %d 因子, %.1f 秒",
            len(result.final_pool),
            result.elapsed_seconds,
        )

        return result

    def _precompute_forward_returns(self, data: pl.DataFrame) -> pl.DataFrame:
        """预计算所有前瞻收益列（避免每个公式重复计算）

        添加列: _fwd_ret_1d, _fwd_ret_5d, _fwd_ret_20d 等
        """
        dc = self.config.date_column
        cc = self.config.code_column
        sorted_df = data.sort([cc, dc])

        for offset in self.config.forward_returns:
            col_name = f"_fwd_ret_{offset}d"
            if col_name not in sorted_df.columns:
                sorted_df = sorted_df.with_columns(
                    (
                        (pl.col("close").shift(-offset).over(cc) - pl.col("close"))
                        / pl.col("close")
                    ).alias(col_name)
                )
                logger.info("[Pipeline] 预计算 %s 完成", col_name)

        return sorted_df

    def _run_alphagpt(self, data: pl.DataFrame) -> Optional[AlphaGptResult]:
        """运行 Alpha-GPT 工作流"""
        try:
            config = AlphaGptConfig(
                objective=self.config.objective,
                iterations=self.config.alphagpt_iterations,
                pool_size=self.config.alphagpt_pool_size,
                top_k=self.config.alphagpt_top_k,
                min_ir_threshold=self.config.min_ir_threshold,
                max_mutual_ic_threshold=self.config.max_mutual_ic,
                forward_returns=list(self.config.forward_returns),
                date_column=self.config.date_column,
                code_column=self.config.code_column,
                llm_provider=self.config.llm_provider,
                llm_model=self.config.llm_model,
                temperature=self.config.temperature,
                temperature_idea_gen=self.config.temperature_idea_gen,
                temperature_formula=self.config.temperature_formula,
                temperature_reflector=self.config.temperature_reflector,
                temperature_critic=self.config.temperature_critic,
            )

            # 构建 LLM 客户端
            llm_client = self._build_llm_client()

            workflow = AlphaGptWorkflow(
                config=config,
                data=data,
                llm_client=llm_client,
            )
            return workflow.run()

        except Exception as e:
            logger.error("[Pipeline] Alpha-GPT 失败: %s", e)
            return None

    def _run_mcts(
        self, data: pl.DataFrame, seed_formulas: Optional[List[str]] = None
    ) -> Optional[MCTSSearchResult]:
        """运行 MCTS 搜索"""
        try:
            config = MCTSSearchConfig(
                iterations=self.config.mcts_iterations,
                max_depth=self.config.mcts_max_depth,
                dedup_threshold=self.config.mcts_dedup_threshold,
                forward_returns=self.config.forward_returns,
                date_column=self.config.date_column,
                code_column=self.config.code_column,
            )

            cache = MCTSCache(MCTSCacheConfig(enabled=True))
            search = MCTSSearch(config=config, cache=cache)

            return search.search(
                data=data,
                seed_formulas=seed_formulas,
                date_column=self.config.date_column,
                code_column=self.config.code_column,
            )

        except Exception as e:
            logger.error("[Pipeline] MCTS 失败: %s", e)
            return None

    def _extract_seed_formulas(
        self, alphagpt_result: Optional[AlphaGptResult]
    ) -> Optional[List[str]]:
        """从 Alpha-GPT 结果提取种子公式"""
        if alphagpt_result is None:
            return None

        formulas = [f.formula for f in alphagpt_result.final_pool]
        logger.info("[Pipeline] 提取 %d 个种子公式", len(formulas))
        return formulas

    def _merge_and_dedup(
        self,
        alphagpt_result: Optional[AlphaGptResult],
        mcts_result: Optional[MCTSSearchResult],
        data: pl.DataFrame,
    ) -> List[FactorMetrics]:
        """合并并去重"""
        all_metrics: List[FactorMetrics] = []

        # 收集 Alpha-GPT 结果
        if alphagpt_result:
            for f in alphagpt_result.final_pool:
                all_metrics.append(FactorMetrics(
                    formula_id=f.formula_id,
                    status="success",
                    ic_mean=f.ic_mean,
                    ir=f.ir,
                    overall_score=f.ir,
                ))

        # 收集 MCTS 结果
        if mcts_result:
            for n in mcts_result.best_k_nodes:
                all_metrics.append(FactorMetrics(
                    formula_id=n.entry_id,
                    status="success",
                    ic_mean=n.metadata.get("ic_mean", 0.0),
                    ir=n.metadata.get("ir", 0.0),
                    overall_score=n.overall_score,
                ))

        if not all_metrics:
            return []

        # 去重
        def get_values(m: FactorMetrics) -> Optional[pl.Series]:
            try:
                # 从 formula_id 提取公式（如果是 FinalFormulaRecord 格式）
                # 或者直接使用 metadata 中的公式
                formula = self._get_formula_from_metrics(m, alphagpt_result, mcts_result)
                if formula:
                    return self.vocab.evaluate(formula, data)
            except Exception:
                pass
            return None

        deduped = deduplicate_mutual_ic(
            all_metrics,
            get_values,
            threshold=self.config.max_mutual_ic,
        )

        # 按 overall_score 降序排序，取 top_k
        deduped.sort(key=lambda m: m.overall_score, reverse=True)
        return deduped[: self.config.top_k]

    def _get_formula_from_metrics(
        self,
        metrics: FactorMetrics,
        alphagpt_result: Optional[AlphaGptResult],
        mcts_result: Optional[MCTSSearchResult],
    ) -> Optional[str]:
        """从 metrics 获取公式字符串"""
        # 从 Alpha-GPT 结果查找
        if alphagpt_result:
            for f in alphagpt_result.final_pool:
                if f.formula_id == metrics.formula_id:
                    return f.formula

        # 从 MCTS 结果查找
        if mcts_result:
            for n in mcts_result.best_k_nodes:
                if n.entry_id == metrics.formula_id:
                    return n.formula

        return None

    def _persist_to_wiki(self, factors: List[FactorMetrics]) -> List[str]:
        """持久化到 Wiki"""
        pages = []
        for f in factors:
            try:
                wiki_factor = self._to_wiki_factor(f)
                page_name = self.wiki.store_factor(wiki_factor)
                pages.append(page_name)
                logger.info("[Pipeline] 保存到 Wiki: %s", page_name)
            except Exception as e:
                logger.warning("[Pipeline] Wiki 保存失败: %s", e)

        return pages

    def _to_wiki_factor(self, metrics: FactorMetrics) -> WikiFactor:
        """转换为 WikiFactor"""
        # 从 metrics 获取公式（如果有）
        formula = self._get_formula_from_metrics(
            metrics,
            None,  # Alpha-GPT 结果不在这里
            None,  # MCTS 结果不在这里
        ) or metrics.formula_id

        return WikiFactor(
            name=metrics.formula_id,
            formula=formula,
            source=FactorSource.AUTO_RESEARCH,
            category=FactorCategory.OTHER,
            tags=["alpha-pipeline", f"ir={metrics.ir:.3f}"],
            ic_mean=metrics.ic_mean,
            ic_std=metrics.ic_std,
            icir=metrics.ir,
            rank_ic_mean=metrics.rank_ic_mean,
            description=f"Auto-mined factor with IR={metrics.ir:.3f}",
        )

    def _build_llm_client(self) -> Any:
        """构建 LLM 客户端"""
        try:
            from QuantNodes.ai.llm.gateway import get_llm_gateway
            return get_llm_gateway()
        except Exception as e:
            logger.warning("[Pipeline] LLM 客户端不可用: %s", e)
            return None

    def _build_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """构建摘要"""
        return {
            "objective": self.config.objective,
            "alphagpt_factors": (
                len(result.alphagpt_result.final_pool)
                if result.alphagpt_result
                else 0
            ),
            "mcts_factors": (
                len(result.mcts_result.best_k_nodes)
                if result.mcts_result
                else 0
            ),
            "final_factors": len(result.final_pool),
            "wiki_pages": len(result.wiki_pages),
            "elapsed_seconds": result.elapsed_seconds,
            "best_ir": (
                max((m.ir for m in result.final_pool), default=0.0)
                if result.final_pool
                else 0.0
            ),
            "avg_ir": (
                sum(m.ir for m in result.final_pool) / len(result.final_pool)
                if result.final_pool
                else 0.0
            ),
        }
