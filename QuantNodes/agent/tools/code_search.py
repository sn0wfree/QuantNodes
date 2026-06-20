# coding=utf-8
"""
代码搜索工具

提供 grep 内容搜索、文件名查找、带上下文的代码搜索。
路径限制在 workspace 内。
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

from .base import Tool
from ._workspace import WorkspaceTool


class CodeSearchTool(WorkspaceTool, Tool):
    """代码搜索工具

    在工作目录中搜索代码内容和文件名。
    """

    MAX_RESULTS = 50
    MAX_CONTEXT_LINES = 3

    def __init__(self, workspace: str | Path):
        super().__init__(workspace)

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return "代码搜索工具：grep 内容搜索、按模式查找文件、带上下文的代码搜索"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["grep", "find_files", "search_code"],
                    "description": "搜索类型",
                },
                "pattern": {
                    "type": "string",
                    "description": "搜索模式（正则表达式或 glob）",
                },
                "path": {
                    "type": "string",
                    "description": "搜索目录（相对于工作目录），默认为整个工作目录",
                },
                "include": {
                    "type": "string",
                    "description": "文件名过滤模式，如 *.py 或 *.{ts,js}",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "上下文行数（search_code 时使用）",
                    "default": 3,
                },
            },
            "required": ["action", "pattern"],
        }

    @property
    def read_only(self) -> bool:
        return True

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve workspace-relative path (Phase J1: now inherited)."""
        return WorkspaceTool._safe_path(self, rel_path)

    def _match_include(self, filename: str, include: str | None) -> bool:
        """检查文件名是否匹配 include 模式"""
        if not include:
            return True
        # 支持 *.py 或 *.{ts,js} 格式
        if "," in include:
            patterns = [p.strip().lstrip("*") for p in include.split(",")]
            return any(filename.endswith(p) for p in patterns)
        ext = include.lstrip("*")
        return filename.endswith(ext)

    async def execute(self, action: str, **kwargs: Any) -> Any:
        return await self._dispatch(action, {
            "grep": self._grep,
            "find_files": self._find_files,
            "search_code": self._search_code,
        }, **kwargs)

    async def _grep(
        self, pattern: str = "", path: str = "", include: str = "", **kw
    ) -> Dict[str, Any]:
        search_dir = self._safe_path(path)
        if not search_dir.exists():
            return {"error": f"Directory not found: {path}"}

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}

        results = []
        for root, dirs, files in os.walk(search_dir):
            # 跳过隐藏目录和 __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for fname in files:
                if not self._match_include(fname, include):
                    continue
                fpath = Path(root) / fname
                if fpath.stat().st_size > 512 * 1024:  # 跳过 > 512KB
                    continue

                try:
                    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            results.append({
                                "file": str(fpath.relative_to(self.workspace)),
                                "line": i,
                                "content": line.rstrip()[:200],
                            })
                            if len(results) >= self.MAX_RESULTS:
                                return {
                                    "results": results,
                                    "truncated": True,
                                    "total": len(results),
                                }
                except (PermissionError, OSError):
                    continue

        return {"results": results, "truncated": False, "total": len(results)}

    async def _find_files(self, pattern: str = "", path: str = "", **kw) -> Dict[str, Any]:
        search_dir = self._safe_path(path)
        if not search_dir.exists():
            return {"error": f"Directory not found: {path}"}

        matches = sorted(search_dir.glob(pattern))
        results = []
        for m in matches:
            if m.is_file():
                results.append({
                    "path": str(m.relative_to(self.workspace)),
                    "size": m.stat().st_size,
                })
                if len(results) >= self.MAX_RESULTS:
                    break

        return {"matches": results, "total": len(results)}

    async def _search_code(
        self, query: str = "", path: str = "", include: str = "",
        context_lines: int = 3, **kw,
    ) -> Dict[str, Any]:
        search_dir = self._safe_path(path)
        if not search_dir.exists():
            return {"error": f"Directory not found: {path}"}

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}

        results = []
        for root, dirs, files in os.walk(search_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for fname in files:
                if not self._match_include(fname, include):
                    continue
                fpath = Path(root) / fname
                if fpath.stat().st_size > 512 * 1024:
                    continue

                try:
                    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context = []
                            for j in range(start, end):
                                prefix = ">>>" if j == i else "   "
                                context.append(f"{prefix} {j + 1}: {lines[j].rstrip()[:150]}")

                            results.append({
                                "file": str(fpath.relative_to(self.workspace)),
                                "line": i + 1,
                                "match": line.rstrip()[:200],
                                "context": "\n".join(context),
                            })
                            if len(results) >= self.MAX_RESULTS:
                                return {
                                    "results": results,
                                    "truncated": True,
                                    "total": len(results),
                                }
                except (PermissionError, OSError):
                    continue

        return {"results": results, "truncated": False, "total": len(results)}
