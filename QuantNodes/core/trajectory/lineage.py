"""谱系追踪: children_of + lineage (从原始到当前)。

提供 stateless 工具函数, 接受 entries dict 或 list 操作。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .entry import TrajectoryEntry


def children_of(
    entries: Mapping[str, TrajectoryEntry] | Iterable[TrajectoryEntry],
    parent_id: str,
) -> list[TrajectoryEntry]:
    """返回指定父辈的所有子代。"""
    items = entries.values() if isinstance(entries, Mapping) else entries
    return [e for e in items if parent_id in e.parent_ids]


def lineage(
    entries: Mapping[str, TrajectoryEntry],
    entry_id: str,
) -> list[TrajectoryEntry]:
    """返回完整谱系 (从原始到当前), 顺序: 最老 → 最新。

    BFS 防环: 同一节点不重复访问, crossover 时只走第一个 parent。
    """
    if entry_id not in entries:
        return []
    chain: list[TrajectoryEntry] = []
    current = entries[entry_id]
    chain.append(current)
    visited = {current.entry_id}
    while current.parent_ids:
        next_id = current.parent_ids[0]
        if next_id in visited or next_id not in entries:
            break
        current = entries[next_id]
        chain.append(current)
        visited.add(current.entry_id)
    chain.reverse()
    return chain


def descendants(
    entries: Mapping[str, TrajectoryEntry],
    entry_id: str,
    max_depth: int | None = None,
) -> list[TrajectoryEntry]:
    """返回所有后代 (子代、孙代...), BFS。"""
    if entry_id not in entries:
        return []
    seen: set[str] = {entry_id}
    queue: list[tuple[TrajectoryEntry, int]] = [(entries[entry_id], 0)]
    result: list[TrajectoryEntry] = []
    while queue:
        node, depth = queue.pop(0)
        if max_depth is not None and depth >= max_depth:
            continue
        for child in children_of(entries, node.entry_id):
            if child.entry_id in seen:
                continue
            seen.add(child.entry_id)
            result.append(child)
            queue.append((child, depth + 1))
    return result
