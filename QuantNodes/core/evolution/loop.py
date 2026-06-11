"""EvolutionLoop — 多轮演化主循环。

调用流程:
    1. round 0: 为每个 direction 调 Hypothesizer.hypothesize()
    2. round 1..N: 调 ParentSelector 选 parent, 再调 Mutator/Crosser 生成子代
    3. 每轮通过 callback (PipelineRunner.run_candidate) 评估
    4. 评估结果写入 TrajectoryPool

设计原则:
    - EvolutionLoop 不直接执行回测, 通过 callback 委托
    - callback 接受 FactorCandidate, 返回 (passed: bool, metrics: dict, feedback: FactorFeedback)
    - 这样 Loop 可与任意 runner / sandbox 配合
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..quality_gate import FactorZoo, QualityGateNode, QualityGateSetting
from ..trajectory import (
    ParentSelector,
    SelectionStrategy,
    TrajectoryEntry,
    TrajectoryPool,
)
from ..knowledge import KnowledgeBase, RAGEvaluator
from .operators import Crosser, FactorCandidate, Hypothesizer, Mutator
from .settings import EvolutionSetting


# 回调签名: candidate -> (passed, metrics, feedback)
EvaluateFn = Callable[[FactorCandidate], tuple[bool, dict, "FactorFeedback"]]


@dataclass
class EvolutionResult:
    """单次演化实验的最终结果。"""
    best_entries: list[TrajectoryEntry] = field(default_factory=list)
    all_entries: list[TrajectoryEntry] = field(default_factory=list)
    rounds_completed: int = 0
    rejected_count: int = 0
    total_count: int = 0


class EvolutionLoop:
    """演化主循环。

    Args:
        settings: EvolutionSetting
        pool: TrajectoryPool (持久化)
        quality_gate: QualityGateNode (可选, pre-backtest 拦截)
        evaluate_fn: 评估回调 (必填) — 接 FactorCandidate, 返回 (passed, metrics, feedback)
    """

    def __init__(
        self,
        settings: EvolutionSetting,
        pool: TrajectoryPool,
        quality_gate: Optional[QualityGateNode] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
        selector: Optional[ParentSelector] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        rag_top_k: int = 3,
        max_ancestor_depth: int = 2,
        max_descendant_depth: int = 2,
        use_compress: bool = False,
        compressor=None,
        rag_evaluator: Optional[RAGEvaluator] = None,
    ):
        self.settings = settings
        self.pool = pool
        self.quality_gate = quality_gate
        self.evaluate_fn = evaluate_fn
        self.selector = selector or ParentSelector(
            strategy=settings.parent_selection_strategy,
            metric=settings.metric,
            top_percent_threshold=settings.top_percent_threshold,
        )
        self.knowledge_base = knowledge_base
        self.rag_top_k = rag_top_k
        self.max_ancestor_depth = max_ancestor_depth
        self.max_descendant_depth = max_descendant_depth
        self.use_compress = use_compress
        self.compressor = compressor
        self.rag_evaluator = rag_evaluator
        self.rag_metrics_history: list[dict] = []  # 每 round 评估结果
        self.hypothesizer = Hypothesizer(
            model=settings.hypothesizer.model,
            max_correction_attempts=settings.hypothesizer.max_correction_attempts,
            seed=settings.hypothesizer.seed,
            knowledge_base=knowledge_base,
            rag_top_k=rag_top_k,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=max_descendant_depth,
            use_compress=use_compress,
            compressor=compressor,
        )
        self.mutator = Mutator(
            model=settings.mutator.model,
            max_correction_attempts=settings.mutator.max_correction_attempts,
            seed=settings.mutator.seed,
        )
        self.crosser = Crosser(
            model=settings.crosser.model,
            max_correction_attempts=settings.crosser.max_correction_attempts,
            seed=settings.crosser.seed,
        )

    def sync_knowledge_base(self) -> int:
        """从 pool 同步未索引 entry 到 KB, 返回新加数。"""
        if self.knowledge_base is None:
            return 0
        return self.knowledge_base.sync_from_pool()

    def _evaluate_rag(
        self,
        round_idx: int,
        directions: list[str],
    ) -> None:
        """RAG 评估: 用 directions 作 query, 评估 Top-K 检索质量。

        简化版 ground truth: 当前 pool 中所有 entry 都视为相关 (实际使用应提供 query→relevant 映射)。
        结果写入 self.rag_metrics_history。
        """
        if not directions or self.rag_evaluator is None:
            return
        # 构造 token_lists: 每 query 取 Top-K 个 entry 的 token 化文本
        retrieved: list[list[str]] = []
        relevant: list[list[str]] = []
        relevance_scores: list[dict[str, float]] = []
        lineage_ids: list[list[str]] = []
        token_lists: list[list[list[str]]] = []

        all_entry_ids = {e.entry_id for e in self.pool.all()}
        for d in directions:
            results = self.knowledge_base.query(d, top_k=self.rag_top_k)
            ids = [e.entry_id for e, _ in results]
            retrieved.append(ids)
            relevant.append(list(all_entry_ids))  # 简化: 全部视为相关
            relevance_scores.append({eid: 1.0 for eid in ids})
            # lineage = 检索结果的 ancestors + descendants
            lin_set: set[str] = set()
            from ..knowledge import expand_lineage
            for eid in ids:
                expanded = expand_lineage(
                    self.pool, eid,
                    max_ancestor_depth=self.max_ancestor_depth,
                    max_descendant_depth=self.max_descendant_depth,
                )
                for _, e in expanded["ancestors"] + expanded["descendants"]:
                    lin_set.add(e.entry_id)
            lineage_ids.append(list(lin_set))
            # tokens: 用 name + hypothesis 简单分词
            tokens_per_entry: list[list[str]] = []
            for e, _ in results:
                cfg = (e.config_snapshot or {}).get("factor", {}) if e else {}
                toks = []
                if cfg.get("name"):
                    toks += cfg["name"].lower().split("_")
                if cfg.get("hypothesis"):
                    toks += cfg["hypothesis"].lower().split()
                tokens_per_entry.append(toks)
            token_lists.append(tokens_per_entry)

        report = self.rag_evaluator.evaluate(
            queries=directions,
            retrieved=retrieved,
            relevant=relevant,
            relevance_scores=relevance_scores,
            lineage_ids=lineage_ids,
            token_lists=token_lists,
        )
        self.rag_metrics_history.append({
            "round": round_idx,
            "n_queries": report.n_queries,
            "hit_at_5": report.hit_at_5,
            "hit_at_10": report.hit_at_10,
            "ndcg_at_5": report.ndcg_at_5,
            "ndcg_at_10": report.ndcg_at_10,
            "mrr": report.mrr,
            "lineage_coverage": report.lineage_coverage,
            "diversity": report.diversity,
        })

    def run(
        self,
        initial_directions: list[str] | None = None,
        initial_candidates: list[FactorCandidate] | None = None,
    ) -> EvolutionResult:
        """执行完整演化循环。

        Args:
            initial_directions: round 0 用 — 调 Hypothesizer 生成的假设列表
            initial_candidates: round 0 用 — 直接提供的 FactorCandidate 列表

        Returns:
            EvolutionResult: 含 best entries + 统计
        """
        if self.evaluate_fn is None:
            raise ValueError("evaluate_fn 不能为 None")

        result = EvolutionResult()
        no_improve_counter = 0
        best_metric_so_far = float("-inf")
        directions = list(initial_directions or [])

        # ------------------------------------------------------------------
        # Round 0: 原始候选
        # ------------------------------------------------------------------
        round0_candidates = self._build_round0(
            initial_directions or [], initial_candidates or [],
        )
        for cand in round0_candidates:
            entry = self._evaluate_and_record(
                cand, operation="original", parent_ids=[],
            )
            result.all_entries.append(entry)
            if entry.feedback and entry.feedback.decision:
                result.total_count += 1
            else:
                result.rejected_count += 1
            best_metric_so_far = _update_best(
                best_metric_so_far, entry, self.settings.metric, no_improve_counter,
            )
        result.rounds_completed = 1

        # ------------------------------------------------------------------
        # Round 1..N: 演化
        # ------------------------------------------------------------------
        for round_idx in range(1, self.settings.max_rounds + 1):
            # 同步 KB (round 0 新增的 entry 可被 round 1 检索到)
            if self.knowledge_base is not None:
                self.knowledge_base.sync_from_pool()
            # RAG 评估 (用 directions 当 query, ground truth = 当前所有 entry)
            if self.rag_evaluator is not None and self.knowledge_base is not None:
                self._evaluate_rag(round_idx, directions or [])
            n_parents = self.settings.parents_per_round
            # crossover 在奇数轮, mutation 在偶数轮 (与文档一致)
            operation = "crossover" if round_idx % 2 == 0 else "mutation"

            if operation == "crossover" and n_parents < 2:
                n_parents = 2

            parents = self.selector.select(self.pool, n=n_parents)
            if not parents:
                break

            parent_candidates = [
                FactorCandidate(
                    factor_id=e.entry_id,
                    name=e.feedback.factor_name if e.feedback else e.entry_id,
                    expression=str(e.config_snapshot.get("factor", {}).get("expression", "")),
                    hypothesis="",
                    description="",
                )
                for e in parents
            ]

            if operation == "mutation" and len(parent_candidates) >= 1:
                child = self.mutator.mutate(parent_candidates[0])
                parent_ids = [parents[0].entry_id]
            elif operation == "crossover" and len(parent_candidates) >= 2:
                child = self.crossover(
                    parent_candidates[0], parent_candidates[1],
                )
                parent_ids = [parents[0].entry_id, parents[1].entry_id]
            else:
                break

            entry = self._evaluate_and_record(
                child, operation=operation, parent_ids=parent_ids,
                round_idx=round_idx,
            )
            result.all_entries.append(entry)
            if entry.feedback and entry.feedback.decision:
                result.total_count += 1
            else:
                result.rejected_count += 1

            new_best, improved = _maybe_update_best(
                best_metric_so_far, entry, self.settings.metric,
            )
            if improved:
                best_metric_so_far = new_best
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            result.rounds_completed = round_idx

            if (
                self.settings.early_stop_patience > 0
                and no_improve_counter >= self.settings.early_stop_patience
            ):
                break

        result.best_entries = self.pool.best(
            top_n=10, metric=self.settings.metric,
        )
        return result

    # ------------------------------------------------------------------
    # 内部: round 0 构建
    # ------------------------------------------------------------------

    def _build_round0(
        self,
        directions: list[str],
        candidates: list[FactorCandidate],
    ) -> list[FactorCandidate]:
        """混合 directions + candidates, 调 Hypothesizer 生成 round 0。"""
        out = list(candidates)
        for d in directions:
            cand = self.hypothesizer.hypothesize(direction=d, description=d)
            out.append(cand)
        return out

    def crossover(
        self,
        parent1: FactorCandidate,
        parent2: FactorCandidate,
    ) -> FactorCandidate:
        """Public 代理, 便于外部直接调用。"""
        return self.crosser.crossover(parent1, parent2)

    # ------------------------------------------------------------------
    # 内部: 评估 + 记录
    # ------------------------------------------------------------------

    def _evaluate_and_record(
        self,
        candidate: FactorCandidate,
        operation: str,
        parent_ids: list[str],
        round_idx: int = 0,
    ) -> TrajectoryEntry:
        """评估 candidate, 写入 TrajectoryPool。"""
        from ..feedback import FactorFeedback

        # Quality gate 短路
        if self.quality_gate is not None:
            gate = self.quality_gate.check({
                "factor_id": candidate.factor_id,
                "name": candidate.name,
                "expression": candidate.expression,
                "hypothesis": candidate.hypothesis,
                "description": candidate.description,
            })
            if not gate["passed"]:
                # REJECTED: 记录但不再 evaluate
                entry = TrajectoryEntry(
                    entry_id=candidate.factor_id,
                    round_idx=round_idx,
                    operation=operation,
                    parent_ids=parent_ids,
                    config_snapshot={
                        "factor": {
                            "name": candidate.name,
                            "expression": candidate.expression,
                            "hypothesis": candidate.hypothesis,
                            "description": candidate.description,
                        },
                    },
                    feedback=gate["feedback"],
                    metrics={},
                )
                self.pool.add(entry)
                return entry

        # 完整评估
        passed, metrics, feedback = self.evaluate_fn(candidate)
        if feedback is None:
            feedback = FactorFeedback(
                factor_id=candidate.factor_id,
                factor_name=candidate.name,
                decision=passed,
                summary="evaluate_fn returned no feedback",
            )

        entry = TrajectoryEntry(
            entry_id=candidate.factor_id,
            round_idx=round_idx,
            operation=operation,
            parent_ids=parent_ids,
            config_snapshot={
                "factor": {
                    "name": candidate.name,
                    "expression": candidate.expression,
                    "hypothesis": candidate.hypothesis,
                    "description": candidate.description,
                },
            },
            feedback=feedback,
            metrics=metrics,
        )
        self.pool.add(entry)
        return entry


def _update_best(
    current_best: float,
    entry: TrajectoryEntry,
    metric: str,
    no_improve_counter: int,
) -> float:
    """更新 best metric (向后兼容, 旧逻辑保留)。"""
    val = float((entry.metrics or {}).get(metric, 0) or 0)
    if val > current_best:
        return val
    return current_best


def _maybe_update_best(
    current_best: float,
    entry: TrajectoryEntry,
    metric: str,
) -> tuple[float, bool]:
    """更新 best, 返回 (new_best, improved)。"""
    val = float((entry.metrics or {}).get(metric, 0) or 0)
    if val > current_best:
        return val, True
    return current_best, False
