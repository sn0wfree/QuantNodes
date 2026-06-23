# coding=utf-8
"""
mcts - QuantAlpha MCTS 因子搜索子包

参考 AlphaJungle 论文思路：
- 把候选因子组织成搜索树
- 每个节点有多维评价
- UCB1 选择策略
- 维度化反馈指导扩展方向
- 频繁子树规避

M2 改进（vs 旧 mcts_search.py）：
- 7 硬编码 EXTENSION_OPS → 26 动态生成（从 OperatorVocab 162 算子）
- 无谱系 → 完整 entry_id + parent_id（可映射 TrajectoryEntry）
- 单一 dimension_scores → 5 通道 FactorFeedback 完整框架
"""

from __future__ import annotations

from QuantNodes.research.quant_alpha.mcts.extension_ops import (
    ExtensionOp,
    ExtensionOpPool,
    DEFAULT_WINDOWS,
)
from QuantNodes.research.quant_alpha.mcts.feedback import (
    MCTSFeedbackConfig,
    collect_all_channels,
    collect_code_channel,
    collect_execution_channel,
    collect_llm_channel,
    collect_shape_channel,
    collect_value_channel,
)
from QuantNodes.research.quant_alpha.mcts.search import (
    MCTSSearch,
    MCTSSearchConfig,
    MCTSSearchResult,
)
from QuantNodes.research.quant_alpha.mcts.tree import (
    MCTSTree,
    MCTSNode,
    NodeStatus,
)

__all__ = [
    # 操作池
    "ExtensionOp",
    "ExtensionOpPool",
    "DEFAULT_WINDOWS",
    # 反馈
    "MCTSFeedbackConfig",
    "collect_all_channels",
    "collect_code_channel",
    "collect_execution_channel",
    "collect_llm_channel",
    "collect_shape_channel",
    "collect_value_channel",
    # 搜索
    "MCTSSearch",
    "MCTSSearchConfig",
    "MCTSSearchResult",
    # 树
    "MCTSTree",
    "MCTSNode",
    "NodeStatus",
]
