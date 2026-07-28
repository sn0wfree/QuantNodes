"""Tests for ContextEngine (Engine B)."""
from pathlib import Path


from quantnodes_strategy_audit.core.lesson import LessonLoader
from quantnodes_strategy_audit.engines.context_engine import ContextEngine
from quantnodes_strategy_audit.engines.static_engine import StaticEngine


class TestContextEngine:
    def test_get_lesson_returns_dict_without_path(
        self, lessons_dir: Path, rules_path: Path
    ):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.get_lesson("L-202")
        assert "id" in result
        assert "title" in result
        assert "check_prompt" in result
        assert "content_markdown" in result
        assert "file_path" not in result  # Q3 = A: don't expose paths

    def test_get_lesson_nonexistent_returns_error(
        self, lessons_dir: Path, rules_path: Path
    ):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.get_lesson("L-999")
        assert "error" in result

    def test_list_lessons_with_filters(self, lessons_dir: Path, rules_path: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.list_lessons(severity="CRITICAL")
        assert result["count"] >= 10
        assert all(lesson["severity"] == "CRITICAL" for lesson in result["lessons"])

    def test_search_lessons(self, lessons_dir: Path, rules_path: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.search_lessons("look-ahead", top_k=5)
        assert "results" in result
        assert result["query"] == "look-ahead"

    def test_get_code_context(self, lessons_dir: Path, rules_path: Path, tmp_path: Path):
        code = """
import pandas as pd


def standardize_v7_10(X):
    '''Standardize v7.10 factor data.'''
    mean = X.mean()
    std = X.std()
    return (X - mean) / std
"""
        f = tmp_path / "v7.py"
        f.write_text(code)
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.get_code_context(
            file=str(f), focus_lines=[5, 6], depth=2
        )
        assert "imports" in result
        assert "enclosing_function" in result
        assert result["enclosing_function"]["name"] == "standardize_v7_10"
        assert "surrounding_lines" in result

    def test_static_precheck(self, lessons_dir: Path, rules_path: Path, tmp_path: Path):
        code = "x = data.shift(-1)\nmean = X.mean()\n"
        f = tmp_path / "test.py"
        f.write_text(code)
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.static_precheck(file=str(f))
        assert result["total_warnings"] >= 2
        assert "by_lesson" in result
        assert "precheck_violated" in result

    def test_submit_finding(self, lessons_dir: Path, rules_path: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        context = ContextEngine(loader, engine)
        result = context.submit_finding({
            "file": "test.py",
            "line": 10,
            "lesson_id": "L-202",
            "status": "VIOLATED",
            "severity": "CRITICAL",
            "evidence": {"snippet": "mean = X.mean()"},
            "fix_suggestion": "use rolling",
            "confidence": 0.95,
        })
        assert result["stored"] is True
        assert result["finding_id"].startswith("F-")
        findings = context.get_findings()
        assert len(findings) == 1

    def test_submit_finding_persists(
        self, lessons_dir: Path, rules_path: Path, tmp_path: Path
    ):
        loader = LessonLoader(builtin_dir=lessons_dir)
        engine = StaticEngine(rules_path)
        log_path = tmp_path / "findings.json"
        context = ContextEngine(loader, engine, findings_log_path=log_path)
        context.submit_finding({
            "file": "test.py",
            "line": 1,
            "lesson_id": "L-202",
            "status": "VIOLATED",
            "severity": "CRITICAL",
        })
        assert log_path.exists()
