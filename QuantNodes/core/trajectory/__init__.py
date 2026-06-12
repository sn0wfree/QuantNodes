"""TrajectoryPool — 演化轨迹池。

公开 API:
    - TrajectoryEntry (dataclass): 单条轨迹
    - Operation (str Enum): 演化操作类型 (M1)
    - TrajectoryPool (class): 池子 + CRUD + 过滤 + 谱系 + 双层持久化
    - ParentSelector (class): 5 种选择策略
    - SelectionStrategy (Enum): 5 种策略枚举
    - children_of / lineage / descendants: 谱系工具函数
"""
from .entry import Operation, TrajectoryEntry
from .pool import TrajectoryPool
from .selector import ParentSelector, SelectionStrategy
from .lineage import children_of, descendants, lineage

__all__ = [
    "TrajectoryEntry",
    "Operation",
    "TrajectoryPool",
    "ParentSelector",
    "SelectionStrategy",
    "children_of",
    "descendants",
    "lineage",
]
