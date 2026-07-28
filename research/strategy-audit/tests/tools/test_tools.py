"""Tests for MCP tools."""
from pathlib import Path

import pytest

from quantnodes_strategy_audit.core.lesson import LessonLoader
from quantnodes_strategy_audit.engines.context_engine import ContextEngine
from quantnodes_strategy_audit.engines.static_engine import StaticEngine
from quantnodes_strategy_audit.tools import (
    get_code_context_tool,
    get_lesson_tool,
    list_lessons_tool,
    search_lessons_tool,
    static_precheck_tool,
    submit_finding_tool,
)


@pytest.fixture
def context_engine(lessons_dir: Path, rules_path: Path) -> ContextEngine:
    loader = LessonLoader(builtin_dir=lessons_dir)
    engine = StaticEngine(rules_path)
    return ContextEngine(loader, engine)


class TestGetLessonTool:
    def test_returns_lesson_dict(self, context_engine: ContextEngine):
        result = get_lesson_tool(context_engine, "L-202")
        assert "id" in result
        assert result["id"] == "L-202"
        assert "check_prompt" in result

    def test_returns_error_for_missing(self, context_engine: ContextEngine):
        result = get_lesson_tool(context_engine, "L-999")
        assert "error" in result


class TestListLessonsTool:
    def test_returns_count_and_list(self, context_engine: ContextEngine):
        result = list_lessons_tool(context_engine)
        assert "count" in result
        assert "lessons" in result
        assert result["count"] >= 40

    def test_filters_work(self, context_engine: ContextEngine):
        result = list_lessons_tool(context_engine, severity="CRITICAL")
        assert all(lesson["severity"] == "CRITICAL" for lesson in result["lessons"])


class TestSearchLessonsTool:
    def test_search_returns_relevant(self, context_engine: ContextEngine):
        result = search_lessons_tool(context_engine, "look-ahead", top_k=3)
        assert result["query"] == "look-ahead"
        assert "results" in result


class TestGetCodeContextTool:
    def test_extracts_context(self, context_engine: ContextEngine, tmp_path: Path):
        code = "import pandas as pd\n\ndef foo():\n    return bar()\n"
        f = tmp_path / "test.py"
        f.write_text(code)
        result = get_code_context_tool(
            context_engine, file=str(f), focus_lines=[4], depth=2
        )
        assert "imports" in result
        assert "enclosing_function" in result


class TestStaticPrecheckTool:
    def test_finds_violations(
        self, context_engine: ContextEngine, tmp_path: Path
    ):
        code = "x = data.shift(-1)\nmean = X.mean()\n"
        f = tmp_path / "test.py"
        f.write_text(code)
        result = static_precheck_tool(context_engine, file=str(f))
        assert result["total_warnings"] >= 2
        assert "precheck_violated" in result


class TestSubmitFindingTool:
    def test_submits_and_returns_id(self, context_engine: ContextEngine):
        result = submit_finding_tool(
            context_engine,
            {
                "file": "test.py",
                "line": 10,
                "lesson_id": "L-202",
                "status": "VIOLATED",
                "severity": "CRITICAL",
                "evidence": {"snippet": "x = data.shift(-1)"},
                "fix_suggestion": "use rolling",
                "confidence": 0.95,
            },
        )
        assert result["stored"] is True
        assert result["finding_id"].startswith("F-")
