# coding=utf-8
"""
Git 操作工具

提供 git status/diff/log/commit 操作。
限制在 workspace 目录内执行。
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from .base import Tool


class GitOpsTool(Tool):
    """Git 操作工具

    安全地执行 git 命令：status、diff、log、commit。
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    @property
    def name(self) -> str:
        return "git_ops"

    @property
    def description(self) -> str:
        return "Git 操作工具：查看状态、差异、提交历史、创建提交"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["git_status", "git_diff", "git_log", "git_commit"],
                    "description": "Git 操作类型",
                },
                "path": {
                    "type": "string",
                    "description": "文件路径（可选，用于 git_diff）",
                },
                "message": {
                    "type": "string",
                    "description": "提交消息（git_commit 时必需）",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要提交的文件列表（可选，默认提交所有变更）",
                },
                "n": {
                    "type": "integer",
                    "description": "显示最近 n 条提交（git_log 时使用）",
                    "default": 10,
                },
            },
            "required": ["action"],
        }

    @property
    def read_only(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def _run_git(self, *args: str) -> Dict[str, Any]:
        """执行 git 命令"""
        cmd = ["git", "-C", str(self.workspace)] + list(args)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
            }
        except asyncio.TimeoutError:
            return {"error": "Git command timed out after 30s"}
        except FileNotFoundError:
            return {"error": "Git is not installed or not in PATH"}

    async def execute(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_log": self._git_log,
            "git_commit": self._git_commit,
        }
        fn = dispatch.get(action)
        if not fn:
            raise ValueError(f"Unknown action: {action}")
        return await fn(**kwargs)

    async def _git_status(self, **kw) -> Dict[str, Any]:
        result = await self._run_git("status", "--porcelain")
        if result.get("error"):
            return result
        if result["returncode"] != 0:
            return {"error": result["stderr"] or "git status failed"}

        lines = result["stdout"].splitlines() if result["stdout"] else []
        files = []
        for line in lines:
            if len(line) >= 3:
                status = line[:2].strip()
                filepath = line[3:]
                files.append({"status": status, "path": filepath})

        return {"clean": len(files) == 0, "files": files, "total": len(files)}

    async def _git_diff(self, path: str = "", **kw) -> Dict[str, Any]:
        args = ["diff"]
        if path:
            args.extend(["--", path])
        result = await self._run_git(*args)
        if result.get("error"):
            return result
        if result["returncode"] != 0:
            return {"error": result["stderr"] or "git diff failed"}

        return {"diff": result["stdout"], "has_changes": bool(result["stdout"])}

    async def _git_log(self, n: int = 10, **kw) -> Dict[str, Any]:
        result = await self._run_git(
            "log", f"-{n}", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=short"
        )
        if result.get("error"):
            return result
        if result["returncode"] != 0:
            return {"error": result["stderr"] or "git log failed"}

        commits = []
        for line in result["stdout"].splitlines():
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append({
                    "hash": parts[0][:8],
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "message": parts[4],
                })

        return {"commits": commits, "total": len(commits)}

    async def _git_commit(self, message: str = "", files: List[str] = None, **kw) -> Dict[str, Any]:
        if not message:
            return {"error": "message is required for git_commit"}

        # Stage files
        if files:
            for f in files:
                result = await self._run_git("add", "--", f)
                if result.get("error"):
                    return result
        else:
            result = await self._run_git("add", "-A")
            if result.get("error"):
                return result

        # Check if there's anything to commit
        status_result = await self._run_git("status", "--porcelain")
        if status_result.get("stdout") == "":
            return {"error": "Nothing to commit (working tree clean)"}

        # Commit
        result = await self._run_git("commit", "-m", message)
        if result.get("error"):
            return result
        if result["returncode"] != 0:
            return {"error": result["stderr"] or "git commit failed"}

        # Get commit hash
        hash_result = await self._run_git("rev-parse", "--short", "HEAD")
        commit_hash = hash_result.get("stdout", "unknown")

        return {
            "status": "ok",
            "commit": commit_hash,
            "message": message,
            "files": files or "all",
        }
