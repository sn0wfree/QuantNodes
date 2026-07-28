"""Tests for Lesson / LessonLoader."""
from pathlib import Path


from quantnodes_strategy_audit.core.lesson import LessonLoader


class TestLessonLoader:
    def test_load_all_loads_48_lessons(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lessons = loader.load_all()
        assert len(lessons) == 48

    def test_lesson_has_required_fields(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lesson = loader.get("L-202")
        assert lesson is not None
        assert lesson.id == "L-202"
        assert lesson.severity == "CRITICAL"
        assert lesson.category == "lookahead"
        assert lesson.auto_checkable == "agent"
        assert "full-sample" in lesson.title.lower() or "前视" in lesson.title

    def test_lesson_check_prompt_extracted(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lesson = loader.get("L-202")
        assert lesson.check_prompt
        assert "StandardScaler" in lesson.check_prompt or "mean" in lesson.check_prompt

    def test_lesson_one_sentence_extracted(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lesson = loader.get("L-101")
        assert lesson.one_sentence
        # one_sentence should mention key concept from the lesson
        assert "Calmar" in lesson.one_sentence or "baseline" in lesson.one_sentence

    def test_get_nonexistent_returns_none(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        assert loader.get("L-999") is None

    def test_list_lessons_by_severity(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        critical = loader.list_lessons(severity="CRITICAL")
        assert all(lesson.severity == "CRITICAL" for lesson in critical)
        assert len(critical) >= 10  # we know there are >= 10 CRITICAL lessons

    def test_list_lessons_by_category(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lookahead = loader.list_lessons(category="lookahead")
        assert all(lesson.category == "lookahead" for lesson in lookahead)
        assert len(lookahead) >= 4

    def test_search_finds_relevant_lessons(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        results = loader.search("全样本 标准化", top_k=3)
        assert len(results) >= 1
        top_lesson, top_score = results[0]
        assert top_score > 0
        # L-202 should be most relevant
        assert top_lesson.id in ("L-202", "L-223", "L-104")

    def test_external_overrides_builtin(self, lessons_dir: Path, tmp_path: Path):
        # Create external lesson that overrides L-202
        external_md = tmp_path / "L-202-override.md"
        external_md.write_text(
            """---
id: L-202
title: OVERRIDE Title
severity: LOW
auto_checkable: manual
category: lookahead
---

# OVERRIDE L-202

## 一句话总结
OVERRIDE summary.
""",
            encoding="utf-8",
        )
        loader = LessonLoader(
            builtin_dir=lessons_dir, external_dirs=[tmp_path]
        )
        lesson = loader.get("L-202")
        assert lesson is not None
        assert lesson.title == "OVERRIDE Title"
        assert lesson.severity == "LOW"

    def test_to_dict_excludes_file_path(self, lessons_dir: Path):
        loader = LessonLoader(builtin_dir=lessons_dir)
        lesson = loader.get("L-202")
        d = lesson.to_dict()
        assert "file_path" not in d
        assert "id" in d
        assert "check_prompt" in d
