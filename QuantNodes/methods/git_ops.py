# coding=utf-8
"""
Git Operations Method

Provides safe Git operations for external agents.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class GitOperationResult:
    status: str
    data: Any = None
    errors: List[str] = field(default_factory=list)


class GitOperations:
    """Safe Git operations for external agents."""

    ALLOWED_REPOS = [
        os.path.expanduser("~/QuantNodes"),
    ]

    FORBIDDEN_COMMANDS = [
        "git push",
        "git stash pop",
        "git reset --hard",
        "git clean",
    ]

    def __init__(self, allowed_repos: List[str] = None):
        self.allowed_repos = allowed_repos or self.ALLOWED_REPOS

    def _is_repo_allowed(self, repo_path: str) -> bool:
        """Check if repository is within allowed directories."""
        abs_path = os.path.abspath(repo_path)
        for allowed in self.allowed_repos:
            if abs_path.startswith(os.path.abspath(allowed)):
                return True
        return False

    def _run_git(self, repo_path: str, *args) -> GitOperationResult:
        """Run a git command in the repository."""
        if not self._is_repo_allowed(repo_path):
            return GitOperationResult(
                status="error",
                errors=[f"Repository not allowed: {repo_path}"]
            )

        cmd = ["git"] + list(args)

        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return GitOperationResult(
                    status="success",
                    data=result.stdout.strip() if result.stdout else None
                )
            else:
                return GitOperationResult(
                    status="error",
                    errors=[result.stderr.strip() if result.stderr else "Git command failed"]
                )
        except subprocess.TimeoutExpired:
            return GitOperationResult(
                status="error",
                errors=["Git command timed out"]
            )
        except Exception as e:
            return GitOperationResult(
                status="error",
                errors=[str(e)]
            )

    def status(self, repo_path: str) -> GitOperationResult:
        """Get git status of repository."""
        return self._run_git(repo_path, "status", "--porcelain")

    def log(
        self,
        repo_path: str,
        max_count: int = 10,
        format: str = "%h %s"
    ) -> GitOperationResult:
        """Get git log entries."""
        return self._run_git(
            repo_path,
            "log",
            f"--max-count={max_count}",
            f"--format={format}"
        )

    def branch_list(self, repo_path: str) -> GitOperationResult:
        """List git branches."""
        return self._run_git(repo_path, "branch", "-a")

    def current_branch(self, repo_path: str) -> GitOperationResult:
        """Get current branch name."""
        return self._run_git(repo_path, "branch", "--show-current")

    def diff(
        self,
        repo_path: str,
        target: str = "HEAD",
        file: str = None
    ) -> GitOperationResult:
        """Get git diff for a target."""
        cmd = ["diff", target]
        if file:
            cmd.append("--")
            cmd.append(file)
        return self._run_git(repo_path, *cmd)

    def show(
        self,
        repo_path: str,
        ref: str,
        file: str = None
    ) -> GitOperationResult:
        """Show file contents at a specific commit."""
        cmd = ["show", ref]
        if file:
            cmd.append("--")
            cmd.append(file)
        return self._run_git(repo_path, *cmd)

    def remote_list(self, repo_path: str) -> GitOperationResult:
        """List git remotes."""
        return self._run_git(repo_path, "remote", "-v")

    def fetch(self, repo_path: str, remote: str = "origin") -> GitOperationResult:
        """Fetch from remote (safe operation)."""
        return self._run_git(repo_path, "fetch", remote)

    def diff_staged(self, repo_path: str, file: str = None) -> GitOperationResult:
        """Get staged changes diff."""
        cmd = ["diff", "--cached"]
        if file:
            cmd.append("--")
            cmd.append(file)
        return self._run_git(repo_path, *cmd)

    def is_clean(self, repo_path: str) -> bool:
        """Check if repository is clean (no uncommitted changes)."""
        result = self.status(repo_path)
        return result.status == "success" and not result.data