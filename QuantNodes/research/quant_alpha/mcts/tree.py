# coding=utf-8
"""
mcts/tree.py - MCTS 搜索树（带谱系追踪）

vs 旧 mcts_search.py:53-62 的 MCTSNode：
- 旧：仅 parent: MCTSNode 引用，无 entry_id
- 新：entry_id (UUID) + parent_id (UUID) + 完整谱系

每节点的 entry_id 可映射到 core/trajectory.TrajectoryEntry
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeStatus(str, Enum):
    """MCTS 节点状态"""
    PENDING = "pending"          # 未评估
    EVALUATED = "evaluated"      # 已评估（5 通道反馈完成）
    PRUNED = "pruned"            # 被质量门拦截
    REJECTED = "rejected"        # 被反馈拒绝


@dataclass
class MCTSNode:
    """MCTS 搜索树节点（带谱系追踪）

    关键字段：
    - entry_id: UUID，唯一标识（可映射到 TrajectoryEntry）
    - parent_id: 父节点 entry_id（None = 根节点）
    - formula: 当前节点公式
    - depth: 树深度（根 = 0）
    - children: 子节点列表
    - visits: 访问次数
    - dimension_scores: 5 通道反馈评分
    - overall_score: 综合评分
    - status: 节点状态
    """
    formula: str
    parent_id: Optional[str] = None
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    depth: int = 0
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    status: NodeStatus = NodeStatus.PENDING
    is_expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 暂存：UCB1 计算时用的父节点引用（避免 O(n) parent_id 查找）
    _parent_ref: Optional["MCTSNode"] = field(default=None, repr=False, compare=False)

    def add_child(self, child: "MCTSNode") -> None:
        """添加子节点（同时设置 parent_id 和 _parent_ref）"""
        child.parent_id = self.entry_id
        child._parent_ref = self
        child.depth = self.depth + 1
        self.children.append(child)
        self.is_expanded = True

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    def ancestors(self) -> List["MCTSNode"]:
        """返回从根到当前节点的祖先链（不含当前节点）"""
        chain = []
        node = self._parent_ref
        while node is not None:
            chain.append(node)
            node = node._parent_ref
        return list(reversed(chain))

    def lineage_depth(self) -> int:
        """返回谱系深度（从根到当前节点的边数）"""
        return self.depth

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """UCB1 评分

        UCB1 = Q(s) + c * sqrt(ln(N_parent) / N_s)

        Returns:
            float('inf') 当 visits = 0（优先扩展）
        """
        if self.visits == 0:
            return float("inf")

        parent = self._parent_ref
        if parent is None or parent.visits == 0:
            return float("inf")

        exploit = self.overall_score
        explore = exploration_weight * math.sqrt(
            math.log(parent.visits) / self.visits
        )
        return exploit + explore

    def __repr__(self) -> str:
        preview = self.formula[:50] + "..." if len(self.formula) > 50 else self.formula
        return (
            f"MCTSNode(entry={self.entry_id[:8]}, "
            f"depth={self.depth}, visits={self.visits}, "
            f"score={self.overall_score:.3f}, "
            f"status={self.status.value}, "
            f"formula={preview!r})"
        )


@dataclass
class MCTSTree:
    """MCTS 搜索树容器

    vs 旧 mcts_search.py：树结构本身相似，但增加：
    - 完整 entry_id 谱系
    - 节点状态追踪
    - 公式缓存去重
    """
    root: MCTSNode = field(default_factory=lambda: MCTSNode(
        formula="__ROOT__", depth=-1,
    ))
    formula_cache: Dict[str, MCTSNode] = field(default_factory=dict)
    total_iterations: int = 0

    def add_node(self, node: MCTSNode, parent: Optional[MCTSNode] = None) -> None:
        """添加节点到树

        Args:
            node: 节点
            parent: 父节点（None = 根节点）
        """
        if parent is None:
            parent = self.root
        parent.add_child(node)
        # 加入公式缓存
        self.formula_cache[node.formula] = node

    def get_by_formula(self, formula: str) -> Optional[MCTSNode]:
        """按公式查找节点（O(1) via cache）"""
        return self.formula_cache.get(formula)

    def all_nodes(self) -> List[MCTSNode]:
        """返回所有节点（DFS）"""
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root:
                result.append(node)
            stack.extend(node.children)
        return result

    def leaves(self) -> List[MCTSNode]:
        """返回所有叶子节点"""
        return [n for n in self.all_nodes() if n.is_leaf()]

    def best_k(self, k: int = 10, metric: str = "overall_score") -> List[MCTSNode]:
        """返回综合评分最高的 k 个节点（按 entry_id 去重）"""
        seen = set()
        unique = []
        nodes = self.all_nodes()
        # 按 metric 降序
        nodes.sort(key=lambda n: getattr(n, metric, 0.0), reverse=True)
        for n in nodes:
            if n.entry_id in seen:
                continue
            seen.add(n.entry_id)
            unique.append(n)
            if len(unique) >= k:
                break
        return unique

    def stats(self) -> Dict[str, Any]:
        """树统计"""
        nodes = self.all_nodes()
        by_status: Dict[str, int] = {}
        for n in nodes:
            s = n.status.value
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total_nodes": len(nodes),
            "by_status": by_status,
            "max_depth": max((n.depth for n in nodes), default=0),
            "total_iterations": self.total_iterations,
        }
