# coding=utf-8
"""
Code Search Method

Provides code search functionality for external agents.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    file_path: str
    line_number: int
    line_content: str
    match_context: str


@dataclass
class CodeSearchResult:
    status: str
    matches: List[SearchResult] = field(default_factory=list)
    total_files: int = 0
    errors: List[str] = field(default_factory=list)


class CodeSearch:
    """Code search functionality for QuantNodes codebase."""

    SEARCH_DIRS = [
        os.path.expanduser("~/QuantNodes"),
        "/tmp/quantnodes",
    ]

    FILE_EXTENSIONS = [".py", ".vue", ".ts", ".js", ".yaml", ".yml", ".json"]

    def __init__(self, search_dirs: List[str] = None):
        self.search_dirs = search_dirs or self.SEARCH_DIRS

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        abs_path = os.path.abspath(path)
        for allowed in self.search_dirs:
            if abs_path.startswith(os.path.abspath(allowed)):
                return True
        return False

    def search(
        self,
        pattern: str,
        file_pattern: str = "*.py",
        case_sensitive: bool = True,
        whole_word: bool = False,
        max_results: int = 100
    ) -> CodeSearchResult:
        """Search for pattern in code files.

        Args:
            pattern: Search pattern (regex supported)
            file_pattern: File pattern to search (e.g., "*.py")
            case_sensitive: Case sensitive search
            whole_word: Match whole word only
            max_results: Maximum number of results

        Returns:
            CodeSearchResult with matches
        """
        result = CodeSearchResult(status="success")

        if whole_word:
            pattern = r'\b' + pattern + r'\b'

        try:
            regex = re.compile(pattern, re.IGNORECASE if not case_sensitive else 0)
        except re.error as e:
            return CodeSearchResult(status="error", errors=[f"Invalid regex: {e}"])

        for search_dir in self.search_dirs:
            if not os.path.isdir(search_dir):
                continue

            for root, dirs, files in os.walk(search_dir):
                if not self._is_path_allowed(root):
                    continue

                for filename in files:
                    if not self._matches_pattern(filename, file_pattern):
                        continue

                    filepath = os.path.join(root, filename)
                    if not self._is_path_allowed(filepath):
                        continue

                    matches = self._search_file(filepath, regex, max_results - len(result.matches))
                    result.matches.extend(matches)

                    if len(result.matches) >= max_results:
                        result.total_files = self._count_files(search_dir)
                        return result

        result.total_files = self._count_files(self.search_dirs)
        return result

    def _search_file(
        self,
        filepath: str,
        regex: re.Pattern,
        max_results: int
    ) -> List[SearchResult]:
        """Search for pattern in a single file."""
        matches = []

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(SearchResult(
                            file_path=filepath,
                            line_number=line_num,
                            line_content=line.strip(),
                            match_context=self._get_context(f, line_num)
                        ))
                        if len(matches) >= max_results:
                            break
        except Exception:
            pass

        return matches

    def _get_context(self, f, line_num: int, context_lines: int = 2) -> str:
        """Get surrounding context lines."""
        try:
            f.seek(0)
            lines = f.readlines()
            start = max(0, line_num - context_lines - 1)
            end = min(len(lines), line_num + context_lines)
            return "".join(lines[start:end]).strip()
        except Exception:
            return ""

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches the pattern."""
        if pattern == "*":
            return True
        pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        try:
            return bool(re.match(pattern, filename, re.IGNORECASE))
        except re.error:
            return filename.endswith(tuple(self.FILE_EXTENSIONS))

    def _count_files(self, dirs: List[str]) -> int:
        """Count searchable files."""
        count = 0
        for search_dir in dirs:
            if not os.path.isdir(search_dir):
                continue
            for root, _, files in os.walk(search_dir):
                if not self._is_path_allowed(root):
                    continue
                for filename in files:
                    if any(filename.endswith(ext) for ext in self.FILE_EXTENSIONS):
                        count += 1
        return count