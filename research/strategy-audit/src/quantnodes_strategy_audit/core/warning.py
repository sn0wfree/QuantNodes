"""Warning data class and Severity enum."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Warning severity levels (matches 05_LESSONS_LIBRARY convention)."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)


@dataclass(frozen=True)
class Warning:
    """Unified warning data structure emitted by all detectors.

    Attributes:
        detector: detector name, e.g. "lookahead.same_period"
        category: detector category, e.g. "lookahead", "nan_safe"
        severity: warning severity
        file: file path (absolute or relative)
        line: 1-indexed line number
        column: 0-indexed column number (optional)
        snippet: code snippet around the warning location
        message: human-readable description
        fix_suggestion: suggested fix (empty string if none)
        rule_url: URL to associated lesson documentation
        related_lessons: list of associated L-NNN lesson IDs
    """

    detector: str
    category: str
    severity: Severity
    file: Path
    line: int
    column: int = 0
    snippet: str = ""
    message: str = ""
    fix_suggestion: str = ""
    rule_url: str = ""
    related_lessons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "detector": self.detector,
            "category": self.category,
            "severity": self.severity.value,
            "file": str(self.file),
            "line": self.line,
            "column": self.column,
            "snippet": self.snippet,
            "message": self.message,
            "fix_suggestion": self.fix_suggestion,
            "rule_url": self.rule_url,
            "related_lessons": list(self.related_lessons),
        }
