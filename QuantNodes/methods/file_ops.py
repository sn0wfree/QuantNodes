# coding=utf-8
"""
File Operations Method

Provides safe file operations for external agents.
"""

import os
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class FileOperationResult:
    status: str
    data: Any = None
    errors: List[str] = field(default_factory=list)


class FileOperations:
    """Safe file operations for external agents."""

    ALLOWED_DIRS = [
        os.path.expanduser("~/QuantNodes"),
        "/tmp/quantnodes",
    ]

    def __init__(self, allowed_dirs: List[str] = None):
        self.allowed_dirs = allowed_dirs or self.ALLOWED_DIRS

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        abs_path = os.path.abspath(path)
        for allowed in self.allowed_dirs:
            if abs_path.startswith(os.path.abspath(allowed)):
                return True
        return False

    def read_file(self, path: str, encoding: str = "utf-8") -> FileOperationResult:
        """Read file contents."""
        if not self._is_path_allowed(path):
            return FileOperationResult(
                status="error",
                errors=[f"Path not allowed: {path}"]
            )

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return FileOperationResult(status="success", data=content)
        except Exception as e:
            return FileOperationResult(status="error", errors=[str(e)])

    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> FileOperationResult:
        """Write content to file."""
        if not self._is_path_allowed(path):
            return FileOperationResult(
                status="error",
                errors=[f"Path not allowed: {path}"]
            )

        try:
            if create_dirs:
                dir_path = os.path.dirname(path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)
            return FileOperationResult(status="success", data={"path": path})
        except Exception as e:
            return FileOperationResult(status="error", errors=[str(e)])

    def list_directory(self, path: str) -> FileOperationResult:
        """List directory contents."""
        if not self._is_path_allowed(path):
            return FileOperationResult(
                status="error",
                errors=[f"Path not allowed: {path}"]
            )

        try:
            entries = os.listdir(path)
            result = []
            for entry in entries:
                full_path = os.path.join(path, entry)
                try:
                    stat = os.stat(full_path)
                    result.append({
                        "name": entry,
                        "is_dir": os.path.isdir(full_path),
                        "size": stat.st_size if os.path.isfile(full_path) else None,
                    })
                except OSError:
                    result.append({"name": entry, "is_dir": False, "size": None})
            return FileOperationResult(status="success", data=result)
        except Exception as e:
            return FileOperationResult(status="error", errors=[str(e)])

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return os.path.isfile(path) and self._is_path_allowed(path)

    def dir_exists(self, path: str) -> bool:
        """Check if directory exists."""
        return os.path.isdir(path) and self._is_path_allowed(path)