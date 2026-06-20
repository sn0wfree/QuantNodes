# coding=utf-8
"""
文件操作工具

提供安全的文件读/写/编辑/列表/glob 操作。
路径限制在 workspace 内，防止 path traversal。
"""

from pathlib import Path
from typing import Any, Dict

from .base import Tool
from QuantNodes.core.path_utils import ensure_parent


class FileOpsTool(Tool):
    """文件操作工具

    安全地读取、写入、编辑文件，列出目录内容，glob 模式匹配。
    所有路径限制在 workspace 内。
    """

    MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def description(self) -> str:
        return (
            "文件操作工具：读/写/编辑文件、列出目录、glob模式匹配。"
            "所有路径限制在工作目录内。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read_file", "write_file", "edit_file", "list_files", "glob_files"],
                    "description": "操作类型",
                },
                "path": {
                    "type": "string",
                    "description": "文件或目录路径（相对于工作目录）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容（write_file 时必需）",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的旧字符串（edit_file 时必需）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新字符串（edit_file 时必需）",
                },
                "pattern": {
                    "type": "string",
                    "description": "glob 模式（glob_files 时必需），如 **/*.py",
                },
                "offset": {
                    "type": "integer",
                    "description": "读取起始行号（从1开始）",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "读取最大行数",
                    "default": 2000,
                },
            },
            "required": ["action"],
        }

    @property
    def read_only(self) -> bool:
        return False

    def _safe_path(self, rel_path: str) -> Path:
        """将相对路径解析为安全的绝对路径"""
        rel_path = rel_path.lstrip("/")
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise ValueError(f"Path traversal not allowed: {rel_path}")
        return target

    async def execute(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_files": self._list_files,
            "glob_files": self._glob_files,
        }
        fn = dispatch.get(action)
        if not fn:
            raise ValueError(f"Unknown action: {action}")
        return await fn(**kwargs)

    async def _read_file(
        self, path: str = "", offset: int = 1, limit: int = 2000, **kw,
    ) -> Dict[str, Any]:
        target = self._safe_path(path)
        if not target.exists():
            return {"error": f"File not found: {path}"}
        if not target.is_file():
            return {"error": f"Not a file: {path}"}
        if target.stat().st_size > self.MAX_FILE_SIZE:
            return {
                "error": (
                    f"File too large ({target.stat().st_size} bytes). "
                    f"Max: {self.MAX_FILE_SIZE}"
                )
            }

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)
        sliced = lines[max(0, offset - 1) : offset - 1 + limit]
        return {
            "content": "".join(sliced),
            "total_lines": total,
            "offset": offset,
            "lines_returned": len(sliced),
        }

    async def _write_file(
        self, path: str = "", content: str = "", **kw,
    ) -> Dict[str, Any]:
        target = self._safe_path(path)
        ensure_parent(target)
        target.write_text(content, encoding="utf-8")
        return {
            "status": "ok",
            "bytes_written": len(content.encode("utf-8")),
            "path": str(target.relative_to(self.workspace)),
        }

    async def _edit_file(
        self, path: str = "", old_string: str = "", new_string: str = "", **kw,
    ) -> Dict[str, Any]:
        if not old_string:
            return {"error": "old_string is required"}
        target = self._safe_path(path)
        if not target.exists():
            return {"error": f"File not found: {path}"}

        original = target.read_text(encoding="utf-8")
        count = original.count(old_string)
        if count == 0:
            return {"error": "old_string not found in file"}
        if count > 1:
            return {
                "error": (
                    f"old_string found {count} times. "
                    "Provide more context to make it unique."
                )
            }

        updated = original.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "status": "ok",
            "replacements": 1,
            "path": str(target.relative_to(self.workspace)),
        }

    async def _list_files(self, path: str = "", pattern: str = "", **kw) -> Dict[str, Any]:
        target = self._safe_path(path or ".")
        if not target.exists():
            return {"error": f"Directory not found: {path}"}
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = []
        for entry in sorted(target.iterdir()):
            rel = str(entry.relative_to(self.workspace))
            entries.append({
                "name": entry.name,
                "path": rel,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return {"entries": entries[:200], "total": len(entries)}

    async def _glob_files(self, pattern: str = "", path: str = "", **kw) -> Dict[str, Any]:
        if not pattern:
            return {"error": "pattern is required for glob_files"}
        base = self._safe_path(path or ".")
        if not base.exists():
            return {"error": f"Directory not found: {path}"}

        matches = sorted(base.glob(pattern))
        results = []
        for m in matches:
            if m.is_file():
                results.append({
                    "path": str(m.relative_to(self.workspace)),
                    "size": m.stat().st_size,
                })
        return {"matches": results[:100], "total": len(results)}
