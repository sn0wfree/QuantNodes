"""谱系展开 — 从根 entry 出发, BFS 收集 ancestors/descendants 到指定深度。

输出格式:
    {
        'root': TrajectoryEntry,
        'ancestors': [(depth, entry), ...],  # depth=1=parent, 2=grandparent, ...
        'descendants': [(depth, entry), ...],
    }
"""
from __future__ import annotations

from collections import deque

from ..trajectory import TrajectoryEntry, TrajectoryPool, children_of


def expand_lineage(
    pool: TrajectoryPool,
    root_id: str,
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 2,
    max_ancestors: int = 8,
    max_descendants: int = 8,
) -> dict:
    """BFS 展开谱系, 返回 root + ancestors + descendants。

    Args:
        pool: TrajectoryPool
        root_id: 中心 entry ID
        max_ancestor_depth: 上溯深度 (1=parent, 2=grandparent, ...)
        max_descendant_depth: 下探深度
        max_ancestors: 最多收集多少 ancestor (避免 token 爆炸)
        max_descendants: 最多收集多少 descendant

    Returns:
        dict: {'root', 'ancestors': [(depth, entry)], 'descendants': [(depth, entry)]}

    H8 (2026-06-20): all_ids set is computed ONCE at function start
    instead of inside every BFS step. Previously each BFS iteration
    re-iterated pool.all() (O(N)) and rebuilt a set, making the total
    cost O(K * N) where K is the BFS depth/width. Now: O(N) total.
    """
    all_ids = {e.entry_id for e in pool.all()}

    if root_id not in all_ids:
        return {"root": None, "ancestors": [], "descendants": []}

    try:
        root = pool.get(root_id)
    except KeyError:
        return {"root": None, "ancestors": [], "descendants": []}

    ancestors: list[tuple[int, TrajectoryEntry]] = []
    descendants: list[tuple[int, TrajectoryEntry]] = []

    # ------------------------------------------------------------------
    # Ancestors: BFS 上溯
    # ------------------------------------------------------------------
    visited_up: set[str] = {root_id}
    queue: deque = deque([(root, 0)])
    while queue and len(ancestors) < max_ancestors:
        node, depth = queue.popleft()
        if depth >= max_ancestor_depth:
            continue
        for pid in node.parent_ids:
            if pid in visited_up or pid not in all_ids:
                continue
            visited_up.add(pid)
            try:
                parent = pool.get(pid)
            except KeyError:
                continue
            ancestors.append((depth + 1, parent))
            queue.append((parent, depth + 1))

    # ------------------------------------------------------------------
    # Descendants: BFS 下探
    # ------------------------------------------------------------------
    visited_down: set[str] = {root_id}
    queue = deque([(root, 0)])
    while queue and len(descendants) < max_descendants:
        node, depth = queue.popleft()
        if depth >= max_descendant_depth:
            continue
        for child in children_of(pool, node.entry_id):
            if child.entry_id in visited_down:
                continue
            visited_down.add(child.entry_id)
            descendants.append((depth + 1, child))
            queue.append((child, depth + 1))

    # 排序: 浅的在前, 同深度内按 round_idx 升序
    ancestors.sort(key=lambda x: (x[0], x[1].round_idx))
    descendants.sort(key=lambda x: (x[0], x[1].round_idx))

    return {
        "root": root,
        "ancestors": ancestors,
        "descendants": descendants,
    }


def expand_lineage_batch(
    pool: TrajectoryPool,
    root_ids: list[str],
    max_ancestor_depth: int = 2,
    max_descendant_depth: int = 2,
    max_ancestors: int = 8,
    max_descendants: int = 8,
) -> list[dict]:
    """批量展开, 去重。"""
    seen: dict[str, dict] = {}
    for rid in root_ids:
        if rid in seen:
            continue
        seen[rid] = expand_lineage(
            pool, rid,
            max_ancestor_depth=max_ancestor_depth,
            max_descendant_depth=max_descendant_depth,
            max_ancestors=max_ancestors,
            max_descendants=max_descendants,
        )
    return list(seen.values())
