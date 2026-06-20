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
from typing import TYPE_CHECKING, Callable, Optional

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

if TYPE_CHECKING:
    from ..feedback import FactorFeedback


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
        workers: int = 1,
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
        self.workers = workers
        self.snapshot_path: str | None = None  # ProcessPool: 预序列化路径
        self.rag_metrics_history: list[dict] = []  # 每 round 评估结果
        self.metric_collector = None  # 延迟注入, 用于 streaming
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

    def _stream_metrics(self, round_idx: int, directions: list[str]) -> None:
        """每轮调用, 更新 metric_collector (如有注入)。"""
        if self.metric_collector is None:
            return
        from ..monitoring import (
            EvolutionMetrics, QualityMetrics, RagMetrics,
        )
        # RAG: 从 rag_metrics_history 提取最新
        for m in self.rag_metrics_history:
            if m.get("round") == round_idx:
                self.metric_collector.add_rag(RagMetrics(
                    round=m.get("round", round_idx),
                    n_queries=m.get("n_queries", 0),
                    hit_at_5=m.get("hit_at_5", 0.0),
                    ndcg_at_5=m.get("ndcg_at_5", 0.0),
                    mrr=m.get("mrr", 0.0),
                    diversity=m.get("diversity", 1.0),
                ))
        # Evolution: 累积统计
        self.metric_collector.update_evolution_from_pool(self.pool, round_idx)
        # Quality: 3 通道
        self.metric_collector.update_quality_from_pool(self.pool, round_idx)

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
        # Round 0: 原始候选 (批量评估, 支持并行)
        # ------------------------------------------------------------------
        round0_candidates = self._build_round0(
            initial_directions or [], initial_candidates or [],
        )
        round0_results = self._batch_evaluate_and_record(
            round0_candidates, round_idx=0,
            ops=["original"] * len(round0_candidates),
            parent_ids_list=[[] for _ in round0_candidates],
        )
        for entry in round0_results:
            result.all_entries.append(entry)
            if entry.feedback and entry.feedback.decision:
                result.total_count += 1
            else:
                result.rejected_count += 1
            best_metric_so_far = _update_best(
                best_metric_so_far, entry, self.settings.metric, no_improve_counter,
            )
        result.rounds_completed = 1
        self._stream_metrics(0, directions or [])

        # ------------------------------------------------------------------
        # Round 1..N: 演化 (workers=1: 串行; workers>1: 并行多候选)
        # ------------------------------------------------------------------
        for round_idx in range(1, self.settings.max_rounds + 1):
            if self.knowledge_base is not None:
                self.knowledge_base.sync_from_pool()
            if self.rag_evaluator is not None and self.knowledge_base is not None:
                self._evaluate_rag(round_idx, directions or [])

            # 生成本轮候选
            round_candidates: list[FactorCandidate] = []
            round_parent_ids_list: list[list[str]] = []
            round_ops: list[str] = []

            # parents_per_round 控制 mutation 父数 (默认 1)
            n_mutation_parents = max(1, self.settings.parents_per_round)
            parents_m = self.selector.select(self.pool, n=n_mutation_parents)
            for pm in parents_m:
                pc = FactorCandidate(
                    factor_id=pm.entry_id,
                    name=pm.feedback.factor_name if pm.feedback else "",
                    expression=str(pm.config_snapshot.get("factor", {}).get("expression", "")),
                )
                child_m = self.mutator.mutate(pc)
                round_candidates.append(child_m)
                round_parent_ids_list.append([pm.entry_id])
                round_ops.append("mutation")

            # crossover 固定需要 2 parents
            parents_x = self.selector.select(self.pool, n=2)
            if len(parents_x) >= 2:
                pcs = [
                    FactorCandidate(
                        factor_id=e.entry_id,
                        name=e.feedback.factor_name if e.feedback else "",
                        expression=str(e.config_snapshot.get("factor", {}).get("expression", "")),
                    )
                    for e in parents_x
                ]
                child_x = self.crossover(pcs[0], pcs[1])
                round_candidates.append(child_x)
                round_parent_ids_list.append([parents_x[0].entry_id, parents_x[1].entry_id])
                round_ops.append("crossover")

            if not round_candidates:
                break

            # 批量评估
            batch_entries = self._batch_evaluate_and_record(
                round_candidates, round_idx,
                ops=round_ops, parent_ids_list=round_parent_ids_list,
            )

            best_in_round = -1.0
            for entry in batch_entries:
                result.all_entries.append(entry)
                if entry.feedback and entry.feedback.decision:
                    result.total_count += 1
                else:
                    result.rejected_count += 1
                m = float((entry.metrics or {}).get(self.settings.metric, 0) or 0)
                if m > best_in_round:
                    best_in_round = m
            if best_in_round > best_metric_so_far:
                best_metric_so_far = best_in_round
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            result.rounds_completed = round_idx

            # Streaming: 每轮更新 MetricCollector (如有注入)
            self._stream_metrics(round_idx, directions)

            if (
                self.settings.early_stop_patience > 0
                and no_improve_counter >= self.settings.early_stop_patience
            ):
                break

        result.best_entries = self.pool.best(
            top_n=self.settings.top_n, metric=self.settings.metric,
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

    def _batch_evaluate_and_record(
        self,
        candidates: list[FactorCandidate],
        round_idx: int = 0,
        ops: list[str] | None = None,
        parent_ids_list: list[list[str]] | None = None,
    ) -> list[TrajectoryEntry]:
        """批量评估 candidate, 写入 pool, 返回 entry 列表。

        workers=1 → 串行; workers>1 → ProcessPoolExecutor 并行。
        Quality gate 在主进程中对每个 candidate 串行检查 (避免跨进程 pickle 问题)。
        """
        n = len(candidates)
        if ops is None:
            ops = ["original"] * n
        if parent_ids_list is None:
            parent_ids_list = [[] for _ in range(n)]

        # 1. Quality gate 短路 (串行, 避免跨进程 pickle)
        valid: list[tuple[int, FactorCandidate, list[str], str]] = []
        for i, (c, op, pids) in enumerate(zip(candidates, ops, parent_ids_list)):
            if self.quality_gate is not None:
                gate = self.quality_gate.check({
                    "factor_id": c.factor_id,
                    "name": c.name,
                    "expression": c.expression,
                    "hypothesis": c.hypothesis,
                    "description": c.description,
                })
                if not gate["passed"]:
                    from ..feedback import FactorFeedback
                    entry = TrajectoryEntry(
                        entry_id=c.factor_id,
                        round_idx=round_idx,
                        operation=op,
                        parent_ids=pids,
                        config_snapshot={"factor": {
                            "name": c.name, "expression": c.expression,
                            "hypothesis": c.hypothesis, "description": c.description,
                        }},
                        feedback=FactorFeedback(
                            factor_id=c.factor_id, factor_name=c.name,
                            decision=False, summary=gate["feedback"].summary,
                        ),
                        metrics={},
                    )
                    self.pool.add(entry)
                    continue
            valid.append((i, c, pids, op))

        # 收集 rejected entries
        rejected_entries: list[TrajectoryEntry] = []
        for i in range(len(candidates)):
            if i not in {v[0] for v in valid}:
                rejected_entries.append(self.pool.get(candidates[i].factor_id))

        # 2. 评估
        entries_map: dict[int, TrajectoryEntry] = {}
        to_eval = [c for _, c, _, _ in valid]
        if not to_eval:
            return rejected_entries

        if self.workers <= 1:
            # 串行: evaluate_fn 返回 tuple (passed, metrics, feedback) 或 dict
            raw_results = [self.evaluate_fn(c) for c in to_eval]
        elif self.snapshot_path is not None:
            # ProcessPool 模式 (真实并行, snapshot 预序列化)
            from ..parallel import parallel_evaluate
            raw_results = parallel_evaluate(
                to_eval, self.evaluate_fn, max_workers=self.workers,
                snapshot_path=self.snapshot_path,
            )
        else:
            # ThreadPool 模式 (无需 pickle, I/O 密集场景)
            from ..parallel import parallel_evaluate, make_worker_evaluate
            worker_fn = make_worker_evaluate(self.evaluate_fn, sleep_ms=0)
            raw_results = parallel_evaluate(
                to_eval, worker_fn, max_workers=self.workers,
            )

        # 统一转为 dict (evaluate_fn 可能返回 tuple 或 dict)
        results_list = []
        for r in raw_results:
            if isinstance(r, dict):
                results_list.append(r)
            elif isinstance(r, tuple) and len(r) >= 3:
                passed, metrics, feedback = r[0], r[1], r[2]
                results_list.append({
                    "passed": bool(passed),
                    "metrics": metrics or {},
                    "feedback_dict": {
                        "factor_id": getattr(feedback, "factor_id", ""),
                        "factor_name": getattr(feedback, "factor_name", ""),
                        "decision": getattr(feedback, "decision", passed),
                        "summary": getattr(feedback, "summary", ""),
                        "metadata": getattr(feedback, "metadata", {}),
                        "channels": {
                            k.value: {"passed": v.passed, "detail": v.detail, "score": v.score}
                            for k, v in getattr(feedback, "channels", {}).items()
                        },
                    } if feedback is not None else None,
                    "error": None,
                })
            else:
                results_list.append({
                    "passed": False, "metrics": {},
                    "feedback_dict": None, "error": str(r),
                })

        for (i, c, pids, op), res in zip(valid, results_list):
            entry = self._make_entry_from_result(
                c, res, operation=op, parent_ids=pids, round_idx=round_idx,
            )
            self.pool.add(entry)
            entries_map[i] = entry

        # 3. 返回全部 entry (包括 quality_gate rejected 的)
        all_entries: list[TrajectoryEntry] = []
        for i in range(len(candidates)):
            if i in entries_map:
                all_entries.append(entries_map[i])
            else:
                # quality_gate rejected → 从 pool 获取
                all_entries.append(self.pool.get(candidates[i].factor_id))
        return all_entries

    def _make_entry_from_result(
        self,
        candidate: FactorCandidate,
        result: dict,
        operation: str,
        parent_ids: list[str],
        round_idx: int = 0,
    ) -> TrajectoryEntry:
        """从 evaluate result dict 构造 TrajectoryEntry。"""
        from ..feedback import FactorFeedback
        passed = bool(result.get("passed", False))
        metrics = result.get("metrics", {})
        feedback_dict = result.get("feedback_dict")
        if feedback_dict is None:
            feedback = FactorFeedback(
                factor_id=candidate.factor_id,
                factor_name=candidate.name,
                decision=passed,
                summary=result.get("error") or "ok",
                metadata=metrics,
            )
        else:
            feedback = FactorFeedback(
                factor_id=feedback_dict.get("factor_id") or candidate.factor_id,
                factor_name=feedback_dict.get("factor_name") or candidate.name,
                decision=feedback_dict.get("decision", passed),
                summary=feedback_dict.get("summary", ""),
                metadata=feedback_dict.get("metadata", {}),
            )
        return TrajectoryEntry(
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
        """评估单个 candidate, 写入 TrajectoryPool (遗留方法, 建议迁移到 _batch_evaluate_and_record)。"""
        results = self._batch_evaluate_and_record(
            [candidate], round_idx=round_idx,
            ops=[operation], parent_ids_list=[parent_ids],
        )
        return results[0]

    def _evaluate_candidate(
        self,
        candidate: FactorCandidate,
    ) -> tuple[bool, dict, FactorFeedback]:
        """EvolutionLoop.evaluate_fn 回调 (PipelineRunner 用)。"""
        from ..feedback import FactorFeedback
        if self.evaluate_fn is not None and self.evaluate_fn is not self._evaluate_candidate:
            return self.evaluate_fn(candidate)
        return False, {}, FactorFeedback(
            factor_id=candidate.factor_id,
            factor_name=candidate.name,
            decision=False, summary="evaluate_fn not set",
        )


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
