"""WorkspaceTool mixin — Phase J1 (2026-06-20).

Consolidates the 4-times-repeated pattern across agent tools:
  - self.workspace = Path(workspace).resolve()
  - _safe_path(rel_path) guard against ../ path traversal

Used by:
  - CodeSearchTool
  - FileOpsTool
  - GitOpsTool
  - TaskTool

Filling the gap: git_ops and task originally did NOT have a _safe_path
guard (relied on Git's own safety or just trusting the input). Now
both inherit WorkspaceTool so they can use the same guard.

The mixin does NOT inherit from Tool — that's the job of the subclass.
Just include it in the inheritance list:
    class CodeSearchTool(WorkspaceTool, Tool):
        ...
"""
from __future__ import annotations

from pathlib import Path


class WorkspaceTool:
    """Mix-in providing workspace path + path-traversal-safe resolver.

    Subclasses MUST call ``super().__init__(workspace)`` or assign
    ``self.workspace`` themselves; the mixin only stores/guards it.
    """

    workspace: Path

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def _safe_path(self, rel_path: str = "") -> Path:
        """Resolve a workspace-relative path, blocking '../' escapes.

        Returns the workspace root when rel_path is empty.
        Raises ValueError if the resolved path is outside the workspace.
        """
        if not rel_path:
            return self.workspace
        rel_path = rel_path.lstrip("/")
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise ValueError(f"Path traversal not allowed: {rel_path}")
        return target
