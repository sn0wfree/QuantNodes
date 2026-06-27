# coding=utf-8
"""
pipeline.py - 端到端因子挖掘流水线（多轮迭代版）

连接 Alpha-GPT → MCTS → 去重 → Wiki，支持多轮迭代和反馈闭环。

Usage::

    from QuantNodes.research.quant_alpha.pipeline import AlphaPipeline, PipelineConfig

    config = PipelineConfig(
        objective="capture A-share reversal effect",
        wiki_path="wiki/",
        alphagpt_iterations=3,
        mcts_iterations=50,
        termination=TerminationConfig(max_rounds=5, target_factors=10),
    )
    pipeline = AlphaPipeline(config)
    result = pipeline.run(data)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.core.feedback import FactorFeedback
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

__all__ = [
    "AlphaPipeline",
    "PipelineConfig",
    "PipelineResult",
    "TerminationConfig",
    "RoundResult",
    "RoundFeedback",
    "EarlyStopping",
]


# ==============================================================================
# 终止条件配置
# ==============================================================================


@dataclass
class TerminationConfig:
    """终止条件配置"""

    # 主要终止条件
    max_rounds: int = 5                     # 最大轮次
    target_factors: int = 10                # 目标因子数量
    min_improvement: float = 0.01           # 最小 IR 提升

    # 早停配置
    early_stopping: bool = True             # 是否启用早停
    patience: int = 3                       # 连续 N 轮无改善则停止

    # 超时配置
    timeout_seconds: int = 3600             # 总超时时间（秒）
    round_timeout_seconds: int = 600        # 单轮超时时间（秒）


# ==============================================================================
# 早停机制
# ==============================================================================


class EarlyStopping:
    """早停机制

    连续 N 轮无改善则停止。
    """

    def __init__(self, patience: int = 3, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_ir: float = 0.0
        self.counter: int = 0

    def should_stop(self, current_ir: float) -> bool:
        """判断是否应该停止

        Args:
            current_ir: 当前轮次的最佳 IR

        Returns:
            True 表示应该停止
        """
        if current_ir > self.best_ir + self.min_improvement:
            self.best_ir = current_ir
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def reset(self) -> None:
        """重置早停状态"""
        self.best_ir = 0.0
        self.counter = 0


# ==============================================================================
# 单轮反馈
# ==============================================================================


@dataclass
class RoundFeedback:
    """单轮反馈"""

    round_num: int
    best_ir: float
    avg_ir: float
    valid_count: int

    # 最佳因子（用于下一轮种子）
    best_formulas: List[str] = field(default_factory=list)

    # 失败模式（避免重复）
    failed_patterns: List[Dict[str, str]] = field(default_factory=list)

    # 改进建议
    suggestions: List[str] = field(default_factory=list)

    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "round_num": self.round_num,
            "best_ir": self.best_ir,
            "avg_ir": self.avg_ir,
            "valid_count": self.valid_count,
            "best_formulas": self.best_formulas,
            "failed_patterns": self.failed_patterns,
            "suggestions": self.suggestions,
            "stats": self.stats,
        }

    def to_markdown(self) -> str:
        """转换为 Markdown 格式（用于注入 Alpha-GPT）"""
        lines = [f"## Round {self.round_num} 反馈", ""]

        # 最佳因子
        lines.append("### 最佳因子（IR >= 0.5）")
        if self.best_formulas:
            for i, formula in enumerate(self.best_formulas[:5], 1):
                lines.append(f"{i}. `{formula}`")
        else:
            lines.append("无")
        lines.append("")

        # 失败模式
        lines.append("### 失败模式（避免重复）")
        if self.failed_patterns:
            for i, pattern in enumerate(self.failed_patterns[:5], 1):
                lines.append(f"{i}. `{pattern['formula']}` - {pattern['reason']}")
        else:
            lines.append("无")
        lines.append("")

        # 改进建议
        lines.append("### 改进建议")
        if self.suggestions:
            for suggestion in self.suggestions:
                lines.append(f"- {suggestion}")
        else:
            lines.append("无")
        lines.append("")

        # 统计信息
        lines.append("### 统计信息")
        lines.append(f"- 最佳 IR: {self.best_ir:.4f}")
        lines.append(f"- 平均 IR: {self.avg_ir:.4f}")
        lines.append(f"- 有效因子: {self.valid_count} 个")
        if self.stats.get("improvement_vs_prev") is not None:
            lines.append(f"- 与上轮对比: IR 提升 {self.stats['improvement_vs_prev']:.4f}")

        return "\n".join(lines)


# ==============================================================================
# 单轮结果
# ==============================================================================


@dataclass
class RoundResult:
    """单轮结果"""

    round_num: int
    alphagpt_result: Optional[AlphaGptResult] = None
    mcts_result: Optional[MCTSSearchResult] = None
    final_pool: List[FactorMetrics] = field(default_factory=list)

    # 反馈信息
    feedback: Optional[RoundFeedback] = None

    # 详细信息
    feedback_details: Dict[str, Any] = field(default_factory=dict)
    mutual_ic_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 统计信息
    elapsed_seconds: float = 0.0


# ==============================================================================
# 流水线配置
# ==============================================================================


@dataclass
class PipelineConfig:
    """流水线配置"""

    # 研究目标
    objective: str

    # Wiki 配置
    wiki_path: str = "wiki/"

    # 终止条件配置
    termination: TerminationConfig = field(default_factory=TerminationConfig)

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
    min_ir_threshold: float = 0.5

    # 评估配置
    min_ic_decay_ratio: float = 0.3
    max_turnover: float = 2.0

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

    # 输出配置
    output_dir: str = "pipeline_output"


# ==============================================================================
# 流水线结果
# ==============================================================================


@dataclass
class PipelineResult:
    """流水线结果"""

    rounds: List[RoundResult] = field(default_factory=list)
    final_pool: List[FactorMetrics] = field(default_factory=list)
    wiki_pages: List[str] = field(default_factory=list)

    # 全局信息
    all_mcts_nodes: List[Any] = field(default_factory=list)
    global_correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 反馈生成
# ==============================================================================


def generate_feedback(
    round_result: RoundResult,
    history: List[RoundResult],
) -> RoundFeedback:
    """生成单轮反馈

    Args:
        round_result: 当前轮次结果
        history: 历史轮次结果

    Returns:
        RoundFeedback
    """
    round_num = round_result.round_num

    # 1. 提取最佳因子
    best_formulas: List[str] = []
    if round_result.mcts_result:
        for node in round_result.mcts_result.valid_nodes:
            ir = abs(node.metadata.get("ir", 0.0))
            if ir >= 0.5:
                best_formulas.append(node.formula)
    best_formulas = best_formulas[:5]

    # 2. 提取失败模式
    failed_patterns: List[Dict[str, str]] = []
    if round_result.mcts_result:
        # 使用 tree.all_nodes() 获取所有节点
        for node in round_result.mcts_result.tree.all_nodes():
            fb = round_result.mcts_result.feedback_cache.get(node.formula)
            if fb and not fb.decision:
                failed_patterns.append({
                    "formula": node.formula,
                    "reason": fb.summary,
                })
    failed_patterns = failed_patterns[:10]

    # 3. 生成改进建议
    suggestions: List[str] = []

    # 分析成功因子的算子
    if best_formulas:
        suggestions.append(f"尝试使用类似 {best_formulas[0][:30]}... 的结构")

    # 分析失败模式
    if failed_patterns:
        reasons = [p["reason"] for p in failed_patterns[:3]]
        suggestions.append(f"避免以下失败模式: {', '.join(reasons)}")

    # 多样性建议
    if len(history) > 0:
        prev_formulas = []
        for h in history:
            if h.feedback:
                prev_formulas.extend(h.feedback.best_formulas)
        if len(set(prev_formulas)) < len(prev_formulas):
            suggestions.append("增加因子多样性，避免重复")

    # 4. 计算统计信息
    best_ir = 0.0
    avg_ir = 0.0
    valid_count = 0

    if round_result.mcts_result:
        valid_nodes = round_result.mcts_result.valid_nodes
        valid_count = len(valid_nodes)
        if valid_nodes:
            irs = [abs(n.metadata.get("ir", 0.0)) for n in valid_nodes]
            best_ir = max(irs)
            avg_ir = sum(irs) / len(irs)

    # 与上轮对比
    improvement_vs_prev = None
    if history and history[-1].feedback:
        prev_best_ir = history[-1].feedback.best_ir
        improvement_vs_prev = best_ir - prev_best_ir

    stats = {
        "improvement_vs_prev": improvement_vs_prev,
        "total_mcts_nodes": (
            len(round_result.mcts_result.tree.all_nodes())
            if round_result.mcts_result
            else 0
        ),
    }

    return RoundFeedback(
        round_num=round_num,
        best_ir=best_ir,
        avg_ir=avg_ir,
        valid_count=valid_count,
        best_formulas=best_formulas,
        failed_patterns=failed_patterns,
        suggestions=suggestions,
        stats=stats,
    )


# ==============================================================================
# 流水线
# ==============================================================================


class AlphaPipeline:
    """端到端因子挖掘流水线（多轮迭代版）

    连接 Alpha-GPT → MCTS → 去重 → Wiki，支持多轮迭代和反馈闭环。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vocab = OperatorVocab.default()
        self.wiki = WikiFactorProxy(config.wiki_path)
        self._start_time: float = 0.0

    def run(self, data: pl.DataFrame) -> PipelineResult:
        """运行流水线

        Args:
            data: 行情数据 DataFrame

        Returns:
            PipelineResult
        """
        self._start_time = time.time()
        result = PipelineResult()

        # 预计算前瞻收益（避免重复计算）
        logger.info("[Pipeline] 预计算前瞻收益...")
        data = self._precompute_forward_returns(data)

        # 初始化早停机制
        early_stopping = EarlyStopping(
            patience=self.config.termination.patience,
            min_improvement=self.config.termination.min_improvement,
        )

        # 多轮迭代
        for round_num in range(1, self.config.termination.max_rounds + 1):
            logger.info("[Pipeline] ====== Round %d 开始 ======", round_num)

            # 检查超时
            if self._check_timeout():
                logger.info("[Pipeline] 超时触发，退出循环")
                break

            # 运行单轮
            round_result = self._run_round(
                round_num, data, result.rounds, early_stopping
            )
            result.rounds.append(round_result)

            # 检查终止条件
            if self._should_stop(
                round_result.feedback, result.final_pool, early_stopping
            ):
                logger.info("[Pipeline] Round %d 触发终止条件", round_num)
                break

            logger.info(
                "[Pipeline] Round %d 完成: IR=%.4f, 有效因子=%d",
                round_num,
                round_result.feedback.best_ir if round_result.feedback else 0.0,
                round_result.feedback.valid_count if round_result.feedback else 0,
            )

        # 最终结果
        result.final_pool = self._select_final_pool(result.rounds)
        result.wiki_pages = self._persist_to_wiki(result.final_pool)
        result.elapsed_seconds = time.time() - self._start_time
        result.summary = self._build_summary(result)

        # 保存结果
        self._save_pipeline_result(result)

        logger.info(
            "[Pipeline] 完成: %d 因子, %.1f 秒",
            len(result.final_pool),
            result.elapsed_seconds,
        )

        return result

    def _run_round(
        self,
        round_num: int,
        data: pl.DataFrame,
        history: List[RoundResult],
        early_stopping: EarlyStopping,
    ) -> RoundResult:
        """运行单轮迭代

        Args:
            round_num: 轮次编号
            data: 行情数据
            history: 历史轮次结果
            early_stopping: 早停机制

        Returns:
            RoundResult
        """
        round_start = time.time()
        round_result = RoundResult(round_num=round_num)

        # Stage 1: Alpha-GPT
        logger.info("[Pipeline] Round %d - Stage 1: Alpha-GPT", round_num)
        feedback = history[-1].feedback if history else None
        round_result.alphagpt_result = self._run_alphagpt(data, feedback)

        # Stage 2: MCTS
        logger.info("[Pipeline] Round %d - Stage 2: MCTS", round_num)
        seed_formulas = self._extract_seed_formulas(round_result.alphagpt_result)
        round_result.mcts_result = self._run_mcts(data, seed_formulas)

        # Stage 3: 合并去重
        logger.info("[Pipeline] Round %d - Stage 3: 合并去重", round_num)
        round_result.final_pool = self._merge_and_dedup(
            round_result.alphagpt_result, round_result.mcts_result, data
        )

        # Stage 4: 生成反馈
        logger.info("[Pipeline] Round %d - Stage 4: 生成反馈", round_num)
        round_result.feedback = generate_feedback(round_result, history)

        # 记录耗时
        round_result.elapsed_seconds = time.time() - round_start

        return round_result

    def _should_stop(
        self,
        feedback: Optional[RoundFeedback],
        final_pool: List[FactorMetrics],
        early_stopping: EarlyStopping,
    ) -> bool:
        """判断是否应该停止

        Args:
            feedback: 当前轮次反馈
            final_pool: 当前因子池
            early_stopping: 早停机制

        Returns:
            True 表示应该停止
        """
        if feedback is None:
            return False

        # 1. 检查目标因子数量
        if len(final_pool) >= self.config.termination.target_factors:
            logger.info("[Pipeline] 达到目标因子数量: %d", len(final_pool))
            return True

        # 2. 检查早停
        if self.config.termination.early_stopping:
            if early_stopping.should_stop(feedback.best_ir):
                logger.info(
                    "[Pipeline] 早停触发: 连续 %d 轮无改善",
                    early_stopping.counter,
                )
                return True

        # 3. 检查超时
        if self._check_timeout():
            logger.info("[Pipeline] 超时触发")
            return True

        return False

    def _check_timeout(self) -> bool:
        """检查是否超时"""
        elapsed = time.time() - self._start_time
        return elapsed > self.config.termination.timeout_seconds

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

    def _run_alphagpt(
        self,
        data: pl.DataFrame,
        feedback: Optional[RoundFeedback] = None,
    ) -> Optional[AlphaGptResult]:
        """运行 Alpha-GPT 工作流

        Args:
            data: 行情数据
            feedback: 上轮反馈（用于注入）

        Returns:
            AlphaGptResult
        """
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

            # 注入反馈
            if feedback is not None:
                config.custom_feedback = feedback.to_markdown()
                logger.info("[Pipeline] 注入 Round %d 反馈到 Alpha-GPT", feedback.round_num)

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

    def _select_final_pool(self, rounds: List[RoundResult]) -> List[FactorMetrics]:
        """从所有轮次中选择最终因子池"""
        all_metrics: List[FactorMetrics] = []

        for round_result in rounds:
            all_metrics.extend(round_result.final_pool)

        if not all_metrics:
            return []

        # 按 IR 降序排序，取 top_k
        all_metrics.sort(key=lambda m: abs(m.ir), reverse=True)
        return all_metrics[: self.config.top_k]

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
        formula = metrics.formula_id  # 简化处理

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

    def _save_pipeline_result(self, result: PipelineResult) -> None:
        """保存流水线结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存每轮结果
        for round_result in result.rounds:
            self._save_round_result(round_result, output_dir)

        # 保存最终结果
        self._save_final_result(result, output_dir)

        logger.info("[Pipeline] 结果已保存到 %s", output_dir)

    def _save_round_result(
        self, round_result: RoundResult, output_dir: Path
    ) -> None:
        """保存单轮结果"""
        round_dir = output_dir / f"round_{round_result.round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 保存反馈信息
        if round_result.feedback:
            feedback_file = round_dir / "feedback.json"
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(round_result.feedback.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存摘要
        summary = {
            "round_num": round_result.round_num,
            "elapsed_seconds": round_result.elapsed_seconds,
            "alphagpt_factors": (
                len(round_result.alphagpt_result.final_pool)
                if round_result.alphagpt_result
                else 0
            ),
            "mcts_factors": (
                len(round_result.mcts_result.valid_nodes)
                if round_result.mcts_result
                else 0
            ),
            "final_factors": len(round_result.final_pool),
            "best_ir": round_result.feedback.best_ir if round_result.feedback else 0.0,
        }
        summary_file = round_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_final_result(
        self, result: PipelineResult, output_dir: Path
    ) -> None:
        """保存最终结果"""
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # 保存最终因子列表
        factors_data = []
        for f in result.final_pool:
            factors_data.append({
                "formula_id": f.formula_id,
                "ir": f.ir,
                "ic_mean": f.ic_mean,
                "overall_score": f.overall_score,
            })
        factors_file = final_dir / "factors.json"
        with open(factors_file, "w", encoding="utf-8") as f:
            json.dump(factors_data, f, indent=2, ensure_ascii=False)

        # 保存报告
        report_lines = [
            "# Alpha Pipeline 最终报告",
            "",
            f"## 目标",
            f"{self.config.objective}",
            "",
            f"## 统计信息",
            f"- 总轮次: {len(result.rounds)}",
            f"- 最终因子数: {len(result.final_pool)}",
            f"- Wiki 页面数: {len(result.wiki_pages)}",
            f"- 总耗时: {result.elapsed_seconds:.1f} 秒",
            "",
            f"## 最终因子",
            "",
            "| 公式ID | IR | IC Mean | Score |",
            "|--------|-----|---------|-------|",
        ]
        for f in result.final_pool:
            report_lines.append(
                f"| {f.formula_id} | {f.ir:.4f} | {f.ic_mean:.4f} | {f.overall_score:.4f} |"
            )
        report_file = final_dir / "report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def _build_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """构建摘要"""
        return {
            "objective": self.config.objective,
            "total_rounds": len(result.rounds),
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
