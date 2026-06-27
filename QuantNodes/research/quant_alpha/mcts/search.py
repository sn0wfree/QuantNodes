# coding=utf-8
"""
mcts/search.py - MCTS 因子搜索主类（UCB1 + 5 通道反馈 + 谱系追踪）

vs 旧 mcts_search.py.MCTSSearch：
- 旧：UCB1 + 7 硬编码 EXTENSION_OPS + 无谱系
- 新：UCB1 + 26 动态操作 + 5 通道反馈 + 谱系追踪 + 公式缓存

返回 MCTSSearchResult 含谱系信息（可映射到 TrajectoryPool）
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.research.quant_alpha.mcts.cache import MCTSCache, MCTSCacheConfig
from QuantNodes.research.quant_alpha.mcts.extension_ops import (
    ExtensionOpPool,
)
from QuantNodes.research.quant_alpha.mcts.feedback import (
    MCTSFeedbackConfig,
    collect_all_channels,
)
from QuantNodes.research.quant_alpha.mcts.tree import (
    MCTSTree,
    MCTSNode,
    NodeStatus,
)
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab

logger = logging.getLogger(__name__)


@dataclass
class MCTSSearchResult:
    """MCTS 搜索结果（含谱系）"""
    tree: MCTSTree
    valid_nodes: List[MCTSNode] = field(default_factory=list)
    best_k_nodes: List[MCTSNode] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    total_iterations: int = 0

    # 统计
    formula_count: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    pruned_count: int = 0

    # 反馈缓存（用于分析失败模式）
    feedback_cache: Dict[str, FactorFeedback] = field(default_factory=dict)


@dataclass
class MCTSSearchConfig:
    """MCTS 搜索配置"""
    iterations: int = 50
    exploration_weight: float = 1.414  # UCB1 c 参数
    max_depth: int = 5  # 树最大深度（防止无限嵌套）
    seed: int = 42
    feedback_config: MCTSFeedbackConfig = field(default_factory=MCTSFeedbackConfig)
    # 控制：是否启用谱系追踪（用于 M3+ TrajectoryPool 集成）
    enable_lineage: bool = True
    # 缓存配置
    cache_config: Optional[MCTSCacheConfig] = None
    # IC/IR 配置
    compute_ic_ir: bool = True
    forward_returns: Tuple[int, ...] = (1, 5, 20)
    date_column: str = "date"
    code_column: str = "code"
    # 互信息去重阈值
    dedup_threshold: float = 0.7
    # PR-5: 结构化逻辑与 LLM 客户端（用于一致性评分）
    structured_logic: Optional[Any] = None
    llm_client: Optional[Any] = None


class MCTSSearch:
    """MCTS 因子搜索（基于 OperatorVocab + 5 通道反馈）"""

    def __init__(
        self,
        vocab: Optional[OperatorVocab] = None,
        op_pool: Optional[ExtensionOpPool] = None,
        config: Optional[MCTSSearchConfig] = None,
        cache: Optional[MCTSCache] = None,
    ):
        self.vocab = vocab or OperatorVocab.default()
        self.op_pool = op_pool or ExtensionOpPool(vocab=self.vocab)
        self.config = config or MCTSSearchConfig()
        self.rng = random.Random(self.config.seed)
        self.cache = cache

        # 统计
        self._tree: Optional[MCTSTree] = None
        self._formula_cache: Dict[str, pl.Series] = {}
        self._feedback_cache: Dict[str, FactorFeedback] = {}

    def search(
        self,
        data: pl.DataFrame,
        seed_formulas: Optional[List[str]] = None,
        date_column: str = "date",
        code_column: str = "code",
        forward_return_column: str = "forward_return",
    ) -> MCTSSearchResult:
        """执行 MCTS 搜索

        Args:
            data: 行情数据
            seed_formulas: 种子公式（可选）
            date_column: 日期列名
            code_column: 股票代码列名
            forward_return_column: 前瞻收益率列名

        Returns:
            MCTSSearchResult（含树、valid_nodes、best_k_nodes）
        """
        start_time = time.time()
        config = self.config

        # 1. 加载共享缓存
        if self.cache is not None:
            self.cache.load(data)
            self._formula_cache = self.cache._formula_cache
            self._feedback_cache = self.cache._feedback_cache

        # 2. 初始化树
        tree = MCTSTree()
        self._tree = tree
        if self.cache is None:
            self._formula_cache.clear()
            self._feedback_cache.clear()

        # 3. 注入种子公式
        available_cols = [
            c for c in data.columns
            if c not in (date_column, code_column, forward_return_column)
        ]
        if seed_formulas is None:
            seed_formulas = self.op_pool.get_seed_formulas(available_cols)

        for formula in seed_formulas:
            if formula in tree.formula_cache:
                continue
            node = MCTSNode(formula=formula, depth=0)
            tree.add_node(node, parent=tree.root)

        # 4. MCTS 主循环
        for iteration in range(config.iterations):
            # 4.1 SELECT: UCB1 选择
            leaf = self._select(tree.root)

            # 4.2 EXPAND: 生成子节点
            if not leaf.is_expanded and leaf.depth < config.max_depth:
                new_node = self._expand(leaf, data, available_cols)
                if new_node is not None:
                    leaf = new_node

            # 4.3 EVALUATE: 5 通道反馈 + IC/IR
            self._evaluate(leaf, data, date_column, forward_return_column)

            # 4.4 BACKUP: 回传评分
            self._backpropagate(leaf)

        # 5. 保存共享缓存
        if self.cache is not None:
            self.cache.save()

        # 6. 收集结果
        tree.total_iterations = config.iterations
        valid_nodes = [
            n for n in tree.all_nodes()
            if n.status == NodeStatus.EVALUATED and n.overall_score > 0
        ]
        best_k = tree.best_k(k=10, metric="overall_score")

        # 7. 互信息去重
        if hasattr(config, 'dedup_threshold') and config.dedup_threshold < 1.0:
            try:
                from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
                    deduplicate_mutual_ic,
                    FactorMetrics,
                )

                def get_values(node: MCTSNode) -> Optional[Any]:
                    try:
                        return self.vocab.evaluate(node.formula, data)
                    except Exception:
                        return None

                # 转换为 FactorMetrics 格式用于去重
                metrics_list = [
                    FactorMetrics(
                        formula_id=n.entry_id,
                        status="success",
                        overall_score=n.overall_score,
                        ic_mean=n.metadata.get("ic_mean", 0.0),
                        ir=n.metadata.get("ir", 0.0),
                    )
                    for n in best_k
                ]

                deduped = deduplicate_mutual_ic(
                    metrics_list,
                    get_values,
                    threshold=config.dedup_threshold,
                )

                # 重建 best_k
                deduped_ids = {m.formula_id for m in deduped}
                best_k = [n for n in best_k if n.entry_id in deduped_ids]

            except Exception as e:
                logger.warning("MCTS 互信息去重失败: %s", e)

        elapsed = time.time() - start_time
        stats = tree.stats()

        return MCTSSearchResult(
            tree=tree,
            valid_nodes=valid_nodes,
            best_k_nodes=best_k,
            elapsed_seconds=elapsed,
            total_iterations=config.iterations,
            formula_count=stats["total_nodes"],
            valid_count=len(valid_nodes),
            rejected_count=stats["by_status"].get("rejected", 0),
            pruned_count=stats["by_status"].get("pruned", 0),
            feedback_cache=self._feedback_cache,
        )

    def _select(self, root: MCTSNode) -> MCTSNode:
        """UCB1 选择最有潜力的叶子节点

        优化：优先选择高 IC/IR 的节点，同时保持探索性。
        """
        node = root
        while node.children:
            # 选择 UCB1 最大的子节点
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                # 优先选择高 IC/IR 的未访问节点
                def _score(n: MCTSNode) -> float:
                    ic_ir = n.metadata.get("ir", 0.0)
                    return abs(ic_ir) if ic_ir != 0.0 else 0.0

                # 50% 概率选择高 IC/IR，50% 概率随机
                if self.rng.random() < 0.5:
                    return max(unvisited, key=_score)
                else:
                    return self.rng.choice(unvisited)
            # 全已访问：选 UCB1 最大
            best = max(node.children, key=lambda c: c.ucb1(self.config.exploration_weight))
            node = best
        return node

    def _expand(
        self,
        node: MCTSNode,
        data: pl.DataFrame,
        available_cols: List[str],
    ) -> Optional[MCTSNode]:
        """扩展节点：生成子公式

        优化：优先使用高 IC/IR 的节点作为父节点，优先选择有效算子。

        Returns:
            新 MCTSNode，None = 公式无效或已存在
        """
        if node.formula == "__ROOT__":
            # 根节点：已经添加种子节点，无需扩展
            return None

        # 随机选一个扩展操作
        try:
            op = self.op_pool.sample()
        except ValueError:
            return None

        # 选窗口
        if op.requires_window:
            w = self.op_pool.sample_window()
        else:
            w = None

        # 选第二输入（仅二元算子）
        if op.max_inputs == 2:
            # 优先选择与当前公式不同的列
            other_cols = [c for c in available_cols if c not in node.formula]
            if other_cols:
                other_col = self.rng.choice(other_cols)
            else:
                other_col = self.rng.choice(available_cols)
            try:
                new_formula = op.template.replace(
                    "{f2}", other_col,
                ).replace("{f}", node.formula)
                if op.requires_window and w is not None:
                    new_formula = new_formula.replace("{w}", str(w))
            except Exception:
                return None
        else:
            try:
                new_formula = op.instantiate(node.formula, w)
            except Exception:
                return None

        # 公式缓存去重
        if new_formula in self._tree.formula_cache:
            return None
        if node.formula.count("(") >= 5 * 2:  # 防止过深嵌套
            return None

        # 快速预检查（避免无效公式进入评估）
        precheck_err = self._precheck_formula(new_formula)
        if precheck_err is not None:
            logger.debug("预检查失败: %s (%s)", new_formula[:50], precheck_err)
            return None

        # 创建子节点
        child = MCTSNode(
            formula=new_formula,
            depth=node.depth + 1,
        )
        node.add_child(child)
        self._tree.add_node(child, parent=node)
        return child

    def _precheck_formula(self, formula: str) -> Optional[str]:
        """快速预检查，返回错误原因或 None

        检查项：
        1. 括号匹配
        2. 长度限制
        3. 未知算子
        """
        # 括号匹配
        if formula.count("(") != formula.count(")"):
            return "bracket mismatch"

        # 长度限制
        if len(formula) > 500:
            return "too long"

        # 未知算子
        import re
        from QuantNodes.research.quant_alpha.llm.parser import ALLOWED_OPERATORS
        ops = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', formula)
        for op in ops:
            if op not in ALLOWED_OPERATORS:
                return f"unknown op: {op}"

        return None

    def _evaluate(
        self,
        node: MCTSNode,
        data: pl.DataFrame,
        date_column: str,
        forward_return_column: str = "forward_return",
    ) -> None:
        """评估节点：5 通道反馈 + IC/IR"""
        if node.status != NodeStatus.PENDING:
            return

        node.status = NodeStatus.EVALUATED
        node.visits += 1

        # 公式缓存
        if node.formula in self._formula_cache:
            result = self._formula_cache[node.formula]
            exception = None
        else:
            result = None
            exception = None
            try:
                result = self.vocab.evaluate(
                    formula=node.formula,
                    data=data,
                    date_column=date_column,
                )
                self._formula_cache[node.formula] = result
            except Exception as e:
                exception = e
                result = None

        # 5+3 通道反馈
        expected_length = len(data)

        # 计算 IC/IR（用于 DECAY 通道）
        ic_metrics = {}
        if self.config.compute_ic_ir and exception is None and result is not None:
            try:
                ic_metrics = self._compute_ic_ir(
                    node.formula, data, date_column, forward_return_column
                )
            except Exception as e:
                logger.debug("IC/IR computation failed for %s: %s", node.formula, e)

        fb = collect_all_channels(
            formula=node.formula,
            result=result,
            expected_length=expected_length,
            config=self.config.feedback_config,
            exception=exception,
            ic_decay=ic_metrics.get("ic_decay"),
            data=data,
            date_column=date_column,
            code_column=self.config.code_column,
            llm_client=self.config.llm_client,
            structured_logic=self.config.structured_logic,
        )
        self._feedback_cache[node.formula] = fb

        # 维度评分
        node.dimension_scores = {
            ch.value: ch_fb.score
            for ch, ch_fb in fb.channels.items()
        }

        # 综合评分 = 所有通道平均 score
        if fb.channels:
            node.overall_score = sum(
                ch.score for ch in fb.channels.values()
            ) / len(fb.channels)
        else:
            node.overall_score = 0.0

        # 决策
        if not fb.decision:
            node.status = NodeStatus.REJECTED

        # 存储 IC/IR 指标（已在前面计算）
        if ic_metrics:
            node.metadata.update(ic_metrics)

        # 元数据
        node.metadata["feedback_summary"] = fb.summary
        node.metadata["decision"] = fb.decision
        if exception:
            node.metadata["exception"] = str(exception)[:200]

    def _compute_ic_ir(
        self,
        formula: str,
        data: pl.DataFrame,
        date_column: str,
        forward_return_column: str,
    ) -> Dict[str, float]:
        """计算 IC/IR 指标"""
        from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool
        import asyncio

        tool = AlphaEvaluateTool()
        result = asyncio.run(
            tool.execute(
                formulas=[formula],
                data=data,
                date_column=date_column,
                forward_returns=list(self.config.forward_returns),
            )
        )
        if result.get("evaluations") and result["evaluations"][0]["status"] == "success":
            metrics = result["evaluations"][0]["metrics"]
            return {
                "ic_mean": metrics.get("ic_mean", 0.0),
                "ic_std": metrics.get("ic_std", 0.0),
                "ir": metrics.get("ir", 0.0),
                "ic_decay": metrics.get("ic_decay", {}),
            }
        return {"ic_mean": 0.0, "ic_std": 0.0, "ir": 0.0}

    def _backpropagate(self, node: MCTSNode) -> None:
        """回传评分：更新所有祖先节点的 overall_score

        优化：使用加权平均而非最大值，考虑 IC/IR 信息。
        """
        current = node
        while current is not None and current.parent_id is not None:
            parent = self._find_parent(current)
            if parent is None:
                break
            # 使用加权平均：IC/IR 越高，权重越大
            if parent.children:
                def _weighted_score(n: MCTSNode) -> float:
                    base = n.overall_score
                    ir = abs(n.metadata.get("ir", 0.0))
                    # IC/IR 越高，权重越大
                    return base * (1.0 + ir)

                best_child = max(
                    parent.children,
                    key=_weighted_score,
                )
                parent.overall_score = _weighted_score(best_child)
            current = parent

    def _find_parent(self, node: MCTSNode) -> Optional[MCTSNode]:
        """通过 parent_id 找到父节点（O(1) via tree index）"""
        if node.parent_id is None:
            return None
        if self._tree is not None:
            return self._tree.get_by_entry_id(node.parent_id)
        # 回退：线性扫描
        for n in self._tree.all_nodes():
            if n.entry_id == node.parent_id:
                return n
        return None

    def get_feedback(self, formula: str) -> Optional[FactorFeedback]:
        """查询公式的 5 通道反馈"""
        return self._feedback_cache.get(formula)

    def stats(self) -> Dict[str, Any]:
        """MCTS 搜索统计"""
        if self._tree is None:
            return {"status": "not_initialized"}
        tree_stats = self._tree.stats()
        return {
            **tree_stats,
            "formula_cache_size": len(self._formula_cache),
            "feedback_cache_size": len(self._feedback_cache),
        }
