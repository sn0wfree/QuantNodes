"""Report generation in multiple formats (text / json / sarif)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from quantnodes_strategy_audit.core.warning import Severity, Warning


class Report:
    """Aggregates Warnings and renders to multiple output formats."""

    def __init__(self, warnings: list[Warning] | None = None):
        self.warnings: list[Warning] = warnings or []

    def add(self, warning: Warning) -> None:
        """Add a warning to the report."""
        self.warnings.append(warning)

    def extend(self, warnings: list[Warning]) -> None:
        """Add multiple warnings."""
        self.warnings.extend(warnings)

    @property
    def summary(self) -> dict:
        """Compute summary statistics."""
        by_severity = Counter(w.severity.value for w in self.warnings)
        by_category = Counter(w.category for w in self.warnings)
        by_detector = Counter(w.detector for w in self.warnings)
        return {
            "total": len(self.warnings),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "by_detector": dict(by_detector),
        }

    def has_critical(self) -> bool:
        """Check if any CRITICAL warnings exist."""
        return any(w.severity == Severity.CRITICAL for w in self.warnings)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "summary": self.summary,
            "warnings": [w.to_dict() for w in self.warnings],
        }

    def to_json(self, indent: int = 2) -> str:
        """Render as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Render as human-readable text."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("QuantNodes Strategy Audit Report")
        lines.append("=" * 60)
        s = self.summary
        lines.append(f"Total warnings: {s['total']}")
        if s["by_severity"]:
            sev_str = ", ".join(f"{k}={v}" for k, v in sorted(s["by_severity"].items()))
            lines.append(f"  By severity: {sev_str}")
        if s["by_category"]:
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted(s["by_category"].items()))
            lines.append(f"  By category: {cat_str}")
        lines.append("")

        # Group by severity (CRITICAL first)
        sorted_warnings = sorted(
            self.warnings,
            key=lambda w: (
                -_severity_rank(w.severity),
                w.file,
                w.line,
            ),
        )
        for w in sorted_warnings:
            lines.append(f"[{w.severity.value}] {w.detector}")
            lines.append(f"  File: {w.file}:{w.line}")
            if w.snippet:
                lines.append(f"  Snippet: {w.snippet}")
            if w.message:
                lines.append(f"  Message: {w.message}")
            if w.fix_suggestion:
                lines.append(f"  Fix: {w.fix_suggestion}")
            if w.related_lessons:
                lessons = ", ".join(w.related_lessons)
                lines.append(f"  Ref: {lessons}")
            if w.rule_url:
                lines.append(f"  URL: {w.rule_url}")
            lines.append("")

        return "\n".join(lines)

    def to_sarif(self, tool_version: str = "0.1.0") -> dict:
        """Render as SARIF 2.1.0 dict (for GitHub Code Scanning)."""
        rules: dict[str, dict] = {}
        results: list[dict] = []

        for w in self.warnings:
            if w.detector not in rules:
                rules[w.detector] = {
                    "id": w.detector,
                    "name": w.detector,
                    "shortDescription": {"text": w.message or w.detector},
                    "fullDescription": {"text": w.detector},
                    "helpUri": w.rule_url or "",
                    "defaultConfiguration": {
                        "level": _severity_to_sarif_level(w.severity)
                    },
                }

            results.append({
                "ruleId": w.detector,
                "level": _severity_to_sarif_level(w.severity),
                "message": {"text": w.message or w.detector},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(w.file)},
                        "region": {
                            "startLine": w.line,
                            "startColumn": w.column + 1 if w.column else 1,
                        },
                    }
                }],
            })

        return {
            "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "quantnodes-strategy-audit",
                        "version": tool_version,
                        "informationUri": "https://github.com/sn0wfree/quantnodes-strategy-audit",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }],
        }

    def write(self, output: Path, fmt: str = "json") -> None:
        """Write report to file."""
        fmt = fmt.lower()
        if fmt == "json":
            content = self.to_json()
        elif fmt == "text":
            content = self.to_text()
        elif fmt == "sarif":
            content = json.dumps(self.to_sarif(), indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {fmt}. Supported: json, text, sarif")
        Path(output).write_text(content, encoding="utf-8")


def _severity_rank(s: Severity) -> int:
    """Higher rank = more severe."""
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}[s.value]


def _severity_to_sarif_level(s: Severity) -> str:
    """Map Severity to SARIF level."""
    return {
        Severity.CRITICAL: "error",
        Severity.HIGH: "error",
        Severity.MEDIUM: "warning",
        Severity.LOW: "note",
        Severity.INFO: "note",
    }[s]
