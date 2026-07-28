"""Engine B: Context provider for Agent.

Provides structured context to Agent via 6 MCP tools:
  1. audit_get_lesson: load lesson markdown
  2. audit_list_lessons: list with filters
  3. audit_get_code_context: AST-based code context
  4. audit_search_lessons: keyword search
  5. audit_static_precheck: run Engine A for specific lessons
  6. audit_submit_finding: Agent submits finding

Engine B does NOT call LLM. It only provides data.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from quantnodes_strategy_audit.core.code_context import CodeContextExtractor
from quantnodes_strategy_audit.core.lesson import LessonLoader
from quantnodes_strategy_audit.engines.static_engine import StaticEngine


@dataclass
class Finding:
    """A finding submitted by Agent."""

    finding_id: str
    file: str
    line: int
    lesson_id: str
    status: str
    severity: str
    evidence: dict = field(default_factory=dict)
    fix_suggestion: str = ""
    confidence: float = 0.0
    submitted_at: str = ""

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "file": self.file,
            "line": self.line,
            "lesson_id": self.lesson_id,
            "status": self.status,
            "severity": self.severity,
            "evidence": self.evidence,
            "fix_suggestion": self.fix_suggestion,
            "confidence": self.confidence,
            "submitted_at": self.submitted_at,
        }


class ContextEngine:
    """Provides structured context to Agent via 6 MCP tools.

    Does NOT call LLM. Does NOT do semantic judgment.
    Only provides:
      - Lesson documents (markdown content)
      - AST-based code context
      - Static Engine A precheck results
      - Finding storage (for Agent to submit)
    """

    def __init__(
        self,
        lesson_loader: LessonLoader,
        static_engine: StaticEngine,
        findings_log_path: Path | None = None,
    ):
        self.lesson_loader = lesson_loader
        self.static_engine = static_engine
        self.code_extractor = CodeContextExtractor()
        self.findings_log_path = Path(findings_log_path) if findings_log_path else None
        self._findings: list[Finding] = []

    def get_lesson(self, lesson_id: str) -> dict[str, Any]:
        """Tool 1: audit_get_lesson."""
        lesson = self.lesson_loader.get(lesson_id)
        if lesson is None:
            return {"error": f"Lesson '{lesson_id}' not found"}
        return lesson.to_dict()

    def list_lessons(
        self,
        category: str | None = None,
        severity: str | None = None,
        auto_checkable: str | None = None,
    ) -> dict[str, Any]:
        """Tool 2: audit_list_lessons."""
        lessons = self.lesson_loader.list_lessons(
            category=category, severity=severity, auto_checkable=auto_checkable
        )
        return {
            "count": len(lessons),
            "lessons": [
                {
                    "id": lesson.id,
                    "title": lesson.title,
                    "severity": lesson.severity,
                    "category": lesson.category,
                    "auto_checkable": lesson.auto_checkable,
                    "one_sentence": lesson.one_sentence,
                    "related_lessons": list(lesson.related_lessons),
                }
                for lesson in lessons
            ],
        }

    def get_code_context(
        self,
        file: str,
        focus_lines: list[int],
        depth: int = 2,
    ) -> dict[str, Any]:
        """Tool 3: audit_get_code_context."""
        context = self.code_extractor.extract(
            Path(file), focus_lines=focus_lines, depth=depth
        )
        return context.to_dict()

    def search_lessons(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Tool 4: audit_search_lessons."""
        results = self.lesson_loader.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "id": lesson.id,
                    "title": lesson.title,
                    "category": lesson.category,
                    "severity": lesson.severity,
                    "relevance": score,
                    "one_sentence": lesson.one_sentence,
                }
                for lesson, score in results
            ],
        }

    def static_precheck(
        self,
        file: str,
        lesson_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Tool 5: audit_static_precheck."""
        warnings = self.static_engine.scan_file(
            Path(file), lesson_ids=lesson_ids
        )
        # Group by lesson
        by_lesson: dict[str, list[dict]] = {}
        for w in warnings:
            for lesson_id in w.related_lessons:
                by_lesson.setdefault(lesson_id, []).append(
                    {
                        "line": w.line,
                        "snippet": w.snippet,
                        "message": w.message,
                        "severity": w.severity.value,
                        "detector": w.detector,
                    }
                )
        return {
            "file": file,
            "total_warnings": len(warnings),
            "by_lesson": by_lesson,
            "precheck_violated": list(by_lesson.keys()),
            "warnings": [w.to_dict() for w in warnings],
        }

    def submit_finding(self, finding_data: dict[str, Any]) -> dict[str, Any]:
        """Tool 6: audit_submit_finding."""
        finding = Finding(
            finding_id=f"F-{uuid.uuid4().hex[:8]}",
            file=finding_data.get("file", ""),
            line=int(finding_data.get("line", 0)),
            lesson_id=finding_data.get("lesson_id", ""),
            status=finding_data.get("status", "UNCLEAR"),
            severity=finding_data.get("severity", "MEDIUM"),
            evidence=finding_data.get("evidence", {}),
            fix_suggestion=finding_data.get("fix_suggestion", ""),
            confidence=float(finding_data.get("confidence", 0.0)),
            submitted_at=datetime.now().isoformat(),
        )
        self._findings.append(finding)
        if self.findings_log_path:
            self._save_findings()
        return {"finding_id": finding.finding_id, "stored": True}

    def get_findings(self) -> list[Finding]:
        """Return all stored findings."""
        return list(self._findings)

    def _save_findings(self) -> None:
        """Persist findings to log file."""
        if not self.findings_log_path:
            return
        self.findings_log_path.parent.mkdir(parents=True, exist_ok=True)
        data = [f.to_dict() for f in self._findings]
        self.findings_log_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
