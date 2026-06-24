# coding=utf-8
"""
MCTS 搜索树 - 蒙特卡洛树搜索因子挖掘

基于 Alpha Jungle 论文思路:
- 把候选因子组织成搜索树
- 每个节点有多维评价
- UCB1 选择策略
- 维度化反馈指导扩展方向
- 频繁子树规避

⚠️ DeprecationWarning (v2.7.0+, since 2026-06-23):
    本模块进入 deprecation 周期。新代码请迁移到
    `QuantNodes.research.quant_alpha.mcts.MCTSSearch` (M2 PR)。

    迁移理由：
    - 7 硬编码扩展操作 → 从 OperatorVocab 动态生成
    - 加 5 通道反馈（execution/shape/code/value/llm）
    - 加谱系追踪（parent_id → entry_id）

    Phase 时间表：
    - Phase A (current): 本文件仍可用，行为完全兼容
    - Phase B (M2+): 本类变 thin wrapper
    - Phase C (v3.0): 归档到 _legacy_3c/
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl

warnings.warn(
    "QuantNodes.research.mcts_search 已弃用 (DeprecationWarning)。"
    "M2 PR 将提供新实现 QuantNodes.research.quant_alpha.mcts.MCTSSearch。",
    DeprecationWarning,
    stacklevel=2,
)

from QuantNodes.research._legacy_3c.factor_evaluator import (
    EvalConfig,
    FactorEvaluationResult,
    FactorEvaluator,
)
from QuantNodes.research._legacy_3c.factor_miner import FactorCandidate


@dataclass
class MCTSNode:
    """MCTS 搜索树节点"""
    formula: str
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)
    visits: int = 0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    is_expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# 扩展操作: (算子模板, 参数范围)
EXTENSION_OPS = [
    # 包裹型: rank(...), zscore(...)
    ("rank({f})", "rank"),
    ("zscore({f})", "zscore"),
    # 窗口型: ts_mean(..., w), ts_std(..., w)
    ("ts_mean({f}, {w})", "ts_mean"),
    ("ts_std({f}, {w})", "ts_std"),
    ("ts_delta({f}, {w})", "ts_delta"),
    # 差值型: {f} - ts_mean({f}, w)
    ("{f} - ts_mean({f}, {w})", "mean_diff"),
    # 比值型: {f} / ts_lag({f}, w)
    ("{f} / ts_lag({f}, {w}) - 1", "return"),
]

WINDOWS = [5, 10, 20, 60]


class MCTSSearch:
    """MCTS 因子搜索

    基于蒙特卡洛树搜索，在因子公式空间中寻找高质量因子。
    """

    def __init__(
        self,
        evaluator: FactorEvaluator = None,
        eval_config: EvalConfig = None,
        exploration_weight: float = 1.414,
        seed: int = 42,
    ):
        self.evaluator = evaluator or FactorEvaluator(eval_config)
        self.config = eval_config or EvalConfig()
        self.exploration_weight = exploration_weight
        self.rng = random.Random(seed)
        self._formula_cache: Dict[str, FactorEvaluationResult] = {}

    def search(
        self,
        data: pl.DataFrame,
        seed_formulas: Optional[List[str]] = None,
        iterations: int = 50,
        date_column: str = "date",
        code_column: str = "code",
        forward_return_column: str = "forward_return",
    ) -> List[FactorEvaluationResult]:
        """执行 MCTS 搜索

        Args:
            data: 行情数据
            seed_formulas: 种子公式 (可选, 从已有因子开始搜索)
            iterations: 迭代次数
            date_column: 日期列名
            code_column: 股票代码列名
            forward_return_column: 前瞻收益率列名

        Returns:
            搜索到的高质量因子列表
        """
        # 创建根节点
        root = MCTSNode(formula="__ROOT__")

        # 添加种子节点
        if seed_formulas:
            for formula in seed_formulas:
                child = MCTSNode(formula=formula, parent=root)
                root.children.append(child)

        # MCTS 主循环
        for _ in range(iterations):
            # 1. SELECT: 选择最有潜力的节点
            node = self._select(root)

            # 2. EXPAND: 生成子节点
            if not node.is_expanded:
                child = self._expand(node, data)
                if child is None:
                    continue
                node = child

            # 3. EVALUATE: 多维度评分
            scores = self._evaluate(
                node, data, date_column, code_column, forward_return_column
            )

            # 4. BACKUP: 回传评分
            self._backpropagate(node, scores)

        # 收集结果
        results = self._collect_results(root)
        return results

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 选择策略"""
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda n: self._ucb1(n))
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        """UCB1 公式"""
        if node.visits == 0:
            return float("inf")

        exploit = node.overall_score
        explore = self.exploration_weight * math.sqrt(
            math.log(node.parent.visits + 1) / node.visits
        )
        return exploit + explore

    def _expand(
        self, node: MCTSNode, data: pl.DataFrame
    ) -> Optional[MCTSNode]:
        """扩展节点: 生成子公式"""
        if node.formula == "__ROOT__":
            # 根节点: 生成种子公式
            seed_formulas = self._generate_seed_formulas(data)
            if not seed_formulas:
                return None
            formula = self.rng.choice(seed_formulas)
        else:
            # 非根节点: 应用扩展操作
            formula = self._apply_extension(node.formula)
            if formula is None:
                return None

        # 检查是否已评估过
        if formula in self._formula_cache:
            return None

        # 检查公式复杂度 (避免过深嵌套)
        depth = formula.count("(") - formula.count(")")
        if depth > 5:
            return None

        child = MCTSNode(formula=formula, parent=node)
        node.children.append(child)
        node.is_expanded = True
        return child

    def _evaluate(
        self,
        node: MCTSNode,
        data: pl.DataFrame,
        date_column: str,
        code_column: str,
        forward_return_column: str,
    ) -> Dict[str, float]:
        """评估节点"""
        # 创建候选因子
        candidate = FactorCandidate(
            name=f"mcts_{hash(node.formula) % 10000:04d}",
            formula=node.formula,
            description=f"MCTS 搜索因子: {node.formula}",
            operators_used=[],
            category=type("C", (), {"value": "other"})(),
            template_name="mcts",
        )

        # 评估
        result = self.evaluator.evaluate(
            candidate=candidate,
            data=data,
            date_column=date_column,
            code_column=code_column,
            forward_return_column=forward_return_column,
        )

        # 缓存结果
        self._formula_cache[node.formula] = result

        # 提取维度评分
        node.dimension_scores = result.dimension_scores
        node.overall_score = result.overall_score
        node.visits += 1

        return result.dimension_scores

    def _backpropagate(self, node: MCTSNode, scores: Dict[str, float]):
        """回传评分到父节点"""
        current = node
        while current.parent is not None:
            current = current.parent
            current.visits += 1

            # 更新父节点的综合评分 (子节点最高分)
            if current.children:
                best_child = max(current.children, key=lambda c: c.overall_score)
                current.overall_score = best_child.overall_score

    def _generate_seed_formulas(self, data: pl.DataFrame) -> List[str]:
        """生成种子公式"""
        available_cols = [
            c for c in data.columns
            if c not in ("date", "code", "forward_return")
        ]

        formulas = []
        for col in available_cols[:5]:  # 最多用5列
            for w in [10, 20, 60]:
                formulas.extend([
                    f"ts_mean({col}, {w})",
                    f"ts_std({col}, {w})",
                    f"ts_delta({col}, {w})",
                    f"rank({col})",
                    f"{col} / ts_lag({col}, {w}) - 1",
                ])

        return formulas

    def _apply_extension(self, formula: str) -> Optional[str]:
        """对公式应用扩展操作"""
        ops = list(EXTENSION_OPS)
        self.rng.shuffle(ops)

        for template, op_name in ops:
            w = self.rng.choice(WINDOWS)
            try:
                new_formula = template.replace("{f}", formula).replace("{w}", str(w))
                # 简单语法检查
                if new_formula.count("(") == new_formula.count(")"):
                    return new_formula
            except Exception:
                continue

        return None

    def _collect_results(
        self, root: MCTSNode, min_score: float = 0.1
    ) -> List[FactorEvaluationResult]:
        """收集搜索结果"""
        results = []
        self._collect_recursive(root, results, min_score)

        # 按综合评分排序
        results.sort(key=lambda r: r.overall_score, reverse=True)

        # 去重
        seen_formulas = set()
        unique = []
        for r in results:
            if r.candidate.formula not in seen_formulas:
                seen_formulas.add(r.candidate.formula)
                unique.append(r)

        return unique

    def _collect_recursive(
        self,
        node: MCTSNode,
        results: List[FactorEvaluationResult],
        min_score: float,
    ):
        """递归收集结果"""
        if node.formula in self._formula_cache:
            result = self._formula_cache[node.formula]
            if result.overall_score > min_score:
                results.append(result)

        for child in node.children:
            self._collect_recursive(child, results, min_score)
