"""Static Engine A: YAML rule-based static analysis.

Runs regex patterns from rules/simple_rules.yaml against Python source code.
Fast, deterministic, no LLM involvement.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import yaml

from quantnodes_strategy_audit.core.warning import Severity, Warning


class StaticEngine:
    """YAML-driven static analyzer.

    Each rule has:
        - id: unique identifier
        - pattern: regex pattern
        - severity: CRITICAL / HIGH / MEDIUM / LOW
        - lesson: L-NNN reference
        - category: classification
        - skip_if_preceded_by / skip_if_followed_by: contextual exclusions
    """

    def __init__(self, rules_path: Path):
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> list[dict]:
        """Load rules from YAML."""
        if not self.rules_path.exists():
            return []
        data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        return data.get("rules", []) if data else []

    def scan_file(
        self,
        file: Path,
        lesson_ids: list[str] | None = None,
        categories: list[str] | None = None,
        severities: list[str] | None = None,
    ) -> list[Warning]:
        """Scan a single file with applicable rules.

        Args:
            file: Python source file
            lesson_ids: Only run rules for these lessons (None = all)
            categories: Only run rules in these categories (None = all)
            severities: Only run rules with these severities (None = all)
        """
        file = Path(file)
        if not file.exists():
            return []

        try:
            code = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []

        warnings: list[Warning] = []
        for rule in self.rules:
            if not self._rule_matches_filter(rule, lesson_ids, categories, severities):
                continue
            for warning in self._apply_rule(file, code, rule):
                warnings.append(warning)
        return warnings

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = "*.py",
        **kwargs,
    ) -> list[Warning]:
        """Scan a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            return []
        files = directory.rglob(pattern) if recursive else directory.glob(pattern)
        warnings: list[Warning] = []
        for file in files:
            if file.is_file():
                warnings.extend(self.scan_file(file, **kwargs))
        return warnings

    def _rule_matches_filter(
        self,
        rule: dict,
        lesson_ids: list[str] | None,
        categories: list[str] | None,
        severities: list[str] | None,
    ) -> bool:
        if lesson_ids and rule.get("lesson") not in lesson_ids:
            return False
        if categories and rule.get("category") not in categories:
            return False
        if severities and rule.get("severity") not in severities:
            return False
        return True

    def _apply_rule(
        self,
        file: Path,
        code: str,
        rule: dict,
    ) -> Iterator[Warning]:
        """Apply a single rule to source code."""
        pattern = rule.get("pattern")
        if not pattern:
            return

        try:
            regex = re.compile(pattern)
        except re.error:
            return

        skip_preceded_by = rule.get("skip_if_preceded_by")
        skip_followed_by = rule.get("skip_if_followed_by")
        skip_window = rule.get("skip_window_lines", 3)
        context_hint = rule.get("context_hint", [])

        for match in regex.finditer(code):
            line_num = code[: match.start()].count("\n") + 1

            # Get the line text of the match for context checking
            lines = code.splitlines()
            lines[line_num - 1] if 0 < line_num <= len(lines) else ""

            # skip_if_preceded_by: check on same line first
            if skip_preceded_by:
                # Build preceding context: same line up to match.start()
                line_start_offset = code.rfind("\n", 0, match.start()) + 1
                preceding_on_line = code[line_start_offset:match.start()]
                # Also check previous N lines
                prev_start = max(0, line_num - 1 - skip_window)
                prev_lines_text = "\n".join(lines[prev_start:line_num - 1])

                if re.search(skip_preceded_by, preceding_on_line + "\n" + prev_lines_text):
                    continue

            if skip_followed_by:
                # Build following context: same line after match.end()
                line_end_offset = code.find("\n", match.end())
                if line_end_offset == -1:
                    line_end_offset = len(code)
                following_on_line = code[match.end():line_end_offset]
                # Also check next N lines
                next_end = min(len(lines), line_num + skip_window)
                next_lines_text = "\n".join(lines[line_num:next_end])

                if re.search(skip_followed_by, following_on_line + "\n" + next_lines_text):
                    continue

            if context_hint and not self._has_context_hint(code, line_num, context_hint):
                continue

            yield self._make_warning(file, line_num, match, rule, code)

    def _has_in_window(
        self,
        code: str,
        pos: int,
        window_size: int,
        pattern: str,
    ) -> bool:
        """Check if pattern appears in code window around position.

        Args:
            code: full source code
            pos: position of match
            window_size: number of characters to check (positive=after, negative=before)
            pattern: regex pattern to search for
        """
        if window_size < 0:
            start = max(0, pos + window_size)
            end = pos
        else:
            start = pos
            end = min(len(code), pos + window_size)
        window = code[start:end]
        return bool(re.search(pattern, window))

    def _has_context_hint(
        self, code: str, line_num: int, hints: list[str], window: int = 10
    ) -> bool:
        """Check if any hint keyword appears in surrounding lines."""
        lines = code.splitlines()
        start = max(0, line_num - window - 1)
        end = min(len(lines), line_num + window)
        surrounding = "\n".join(lines[start:end]).lower()
        return any(hint.lower() in surrounding for hint in hints)

    def _make_warning(
        self,
        file: Path,
        line_num: int,
        match: re.Match,
        rule: dict,
        code: str,
    ) -> Warning:
        """Create a Warning from a rule match."""
        lines = code.splitlines()
        lines[line_num - 1] if 0 < line_num <= len(lines) else ""
        start_line = max(0, line_num - 1)
        end_line = min(len(lines), line_num + 1)
        context = "\n".join(lines[start_line:end_line])

        return Warning(
            detector=rule.get("id", "unknown"),
            category=rule.get("category", "unknown"),
            severity=Severity(rule.get("severity", "MEDIUM")),
            file=file,
            line=line_num,
            snippet=context,
            message=rule.get("description", ""),
            fix_suggestion=self._generate_fix_suggestion(rule),
            rule_url=f"https://docs.quant-audit.dev/lessons/{rule.get('lesson', '')}",
            related_lessons=(rule.get("lesson", ""),),
        )

    def _generate_fix_suggestion(self, rule: dict) -> str:
        """Generate a fix suggestion based on rule type."""
        rule.get("id", "")
        lesson = rule.get("lesson", "")
        return f"参考教训 {lesson} ({rule.get('description', '')})"
