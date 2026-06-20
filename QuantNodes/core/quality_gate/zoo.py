"""FactorZoo — 历史通过因子的 AST hash 库。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterator

import pandas as pd
from QuantNodes.core.path_utils import ensure_parent


def ast_hash(expression: str) -> int:
    """AST 规范化 hash (跨进程确定)。

    使用 `hashlib.sha256` 替代 Python 内置 `hash()` (后者受 `PYTHONHASHSEED` 影响,
    跨进程不幂等, 会导致 ProcessPool 模式下的 redundancy check 静默失效)。

    注: `ast.dump(annotate_fields=False)` 移除字段名 ('id', 'arg'),
    只保留节点类型和基本字面量, 实现"结构等价即同 hash"。
    """
    tree = ast.parse(expression)
    payload = ast.dump(tree, annotate_fields=False).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class FactorZoo:
    """因子 Zoo — 存储历史通过因子的 AST hash。

    Args:
        path: Parquet 持久化路径 (None=纯内存, 不持久化)
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else None
        self._entries: dict[int, str] = {}  # hash -> expression
        if self.path is not None and self.path.exists():
            self._load()

    def add(self, expression: str) -> int:
        """添加一个因子, 返回其 hash; 重复 hash 不重复添加。"""
        h = ast_hash(expression)
        if h not in self._entries:
            self._entries[h] = expression
            self._save()
        return h

    def contains(self, expression: str) -> bool:
        """检查表达式是否已在 Zoo 中 (完全相同 AST 结构)。"""
        return ast_hash(expression) in self._entries

    def hamming_to(self, expression: str) -> list[tuple[int, int, str]]:
        """返回与 Zoo 中所有 hash 的汉明距离列表 [(dist, hash, expr), ...]。"""
        new_h = ast_hash(expression)
        results = []
        for h, expr in self._entries.items():
            dist = bin(new_h ^ h).count("1")
            results.append((dist, h, expr))
        results.sort(key=lambda x: x[0])
        return results

    def min_hamming(self, expression: str) -> int:
        """返回与 Zoo 中最近因子的汉明距离 (Zoo 为空时返回 +inf)。"""
        if not self._entries:
            return float("inf")  # type: ignore[return-value]
        return self.hamming_to(expression)[0][0]

    def clear(self) -> None:
        """清空 Zoo (谨慎)。"""
        self._entries.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return iter(self._entries.items())

    def _save(self) -> None:
        if self.path is None or not self._entries:
            return
        ensure_parent(self.path)
        df = pd.DataFrame(
            [(h, e) for h, e in self._entries.items()],
            columns=["hash", "expression"],
        )
        df.to_parquet(self.path, index=False)

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            df = pd.read_parquet(self.path)
        except Exception:
            return
        for _, row in df.iterrows():
            self._entries[int(row["hash"])] = str(row["expression"])
