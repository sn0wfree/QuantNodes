"""Lesson data class and loader.

Loads L-NNN lessons from markdown files in the lessons/ directory.
Each markdown file has YAML frontmatter (id, severity, etc.) and a check_prompt section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Lesson:
    """A lesson loaded from markdown.

    Attributes:
        id: L-NNN identifier (e.g., "L-202")
        title: Short title
        severity: CRITICAL / HIGH / MEDIUM / LOW
        auto_checkable: static / agent / partial / manual
        category: Category classification
        related_lessons: List of related lesson IDs
        related_daily: List of daily lesson IDs (e.g., "L-20260715-1")
        source: Source file (e.g., "05_LESSONS_LIBRARY.md")
        file_path: Path to the markdown file (absolute)
        content_markdown: Full markdown content
        check_prompt: Extracted check prompt for agents
        one_sentence: One-sentence summary
    """

    id: str
    title: str
    severity: str
    auto_checkable: str
    category: str
    related_lessons: tuple[str, ...] = ()
    related_daily: tuple[str, ...] = ()
    source: str = ""
    file_path: Path = field(default_factory=Path)
    content_markdown: str = ""
    check_prompt: str = ""
    one_sentence: str = ""

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict.

        Note: file_path is NOT exposed in dict (Q3 = A: don't expose paths).
        """
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "auto_checkable": self.auto_checkable,
            "category": self.category,
            "related_lessons": list(self.related_lessons),
            "related_daily": list(self.related_daily),
            "source": self.source,
            "one_sentence": self.one_sentence,
            "check_prompt": self.check_prompt,
            "content_markdown": self.content_markdown,
        }


class LessonLoader:
    """Load lessons from one or more directories.

    Lesson sources are merged; later sources override earlier ones by ID.
    """

    def __init__(
        self,
        builtin_dir: Path,
        external_dirs: list[Path] | None = None,
    ):
        self.builtin_dir = Path(builtin_dir)
        self.external_dirs = [Path(d) for d in (external_dirs or [])]

    def load_all(self) -> dict[str, Lesson]:
        """Load all lessons from all sources.

        Returns: dict mapping lesson_id -> Lesson.
        External sources override builtin.
        """
        lessons: dict[str, Lesson] = {}
        for source_dir in [self.builtin_dir] + self.external_dirs:
            if not source_dir.exists():
                continue
            for md_file in sorted(source_dir.glob("L-*.md")):
                try:
                    lesson = self._parse_markdown(md_file)
                    lessons[lesson.id] = lesson
                except Exception:
                    continue
        return lessons

    def get(self, lesson_id: str) -> Lesson | None:
        """Get lesson by ID."""
        return self.load_all().get(lesson_id)

    def list_lessons(
        self,
        category: str | None = None,
        severity: str | None = None,
        auto_checkable: str | None = None,
    ) -> list[Lesson]:
        """List lessons with optional filters."""
        lessons = list(self.load_all().values())
        if category:
            lessons = [lesson for lesson in lessons if lesson.category == category]
        if severity:
            lessons = [lesson for lesson in lessons if lesson.severity == severity]
        if auto_checkable:
            lessons = [
                lesson for lesson in lessons if lesson.auto_checkable == auto_checkable
            ]
        return sorted(lessons, key=lambda lesson: lesson.id)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Lesson, float]]:
        """Search lessons by keyword relevance.

        Simple BM25-like scoring based on keyword frequency.
        """
        query_tokens = set(query.lower().split())
        lessons = list(self.load_all().values())
        scored = []
        for lesson in lessons:
            text = f"{lesson.title} {lesson.one_sentence} {lesson.content_markdown}".lower()
            text_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                score = overlap / (1 + len(text_tokens))
                scored.append((lesson, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _parse_markdown(self, file_path: Path) -> Lesson:
        """Parse a markdown lesson file with YAML frontmatter."""
        content = file_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"No YAML frontmatter in {file_path}")

        fm_text = frontmatter_match.group(1)
        body = frontmatter_match.group(2).strip()

        # Simple YAML parsing (key: value)
        fm_dict: dict[str, str | list[str]] = {}
        current_key: str | None = None
        current_list: list[str] | None = None
        for line in fm_text.split("\n"):
            if line.startswith("  - "):
                if current_list is not None:
                    current_list.append(line.strip()[2:].strip())
                continue
            if ":" in line:
                if current_list is not None and current_key:
                    fm_dict[current_key] = current_list
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    items = [x.strip().strip("\"'") for x in value[1:-1].split(",") if x.strip()]
                    fm_dict[key] = items
                    current_list = None
                    current_key = None
                elif value == "":
                    current_key = key
                    current_list = []
                else:
                    fm_dict[key] = value.strip("\"'")
                    current_list = None
                    current_key = None
        if current_list is not None and current_key:
            fm_dict[current_key] = current_list

        # Extract check_prompt section
        check_prompt = self._extract_section(body, "检测 prompt")
        if not check_prompt:
            check_prompt = self._extract_section(body, "检测清单")

        # Extract one sentence
        one_sentence = ""
        for line in body.split("\n"):
            if line.startswith("## 一句话总结"):
                idx = body.find("## 一句话总结")
                next_section = body.find("\n## ", idx + 1)
                if next_section == -1:
                    next_section = len(body)
                one_sentence = body[idx + len("## 一句话总结"):next_section].strip()
                break

        return Lesson(
            id=str(fm_dict.get("id", file_path.stem)),
            title=str(fm_dict.get("title", "")),
            severity=str(fm_dict.get("severity", "MEDIUM")),
            auto_checkable=str(fm_dict.get("auto_checkable", "manual")),
            category=str(fm_dict.get("category", "")),
            related_lessons=tuple(fm_dict.get("related_lessons", []) or []),
            related_daily=tuple(fm_dict.get("related_daily", []) or []),
            source=str(fm_dict.get("source", "")),
            file_path=file_path.resolve(),
            content_markdown=body,
            check_prompt=check_prompt,
            one_sentence=one_sentence,
        )

    def _extract_section(self, body: str, section_title: str) -> str:
        """Extract a markdown section's content."""
        idx = body.find(f"## {section_title}")
        if idx == -1:
            return ""
        next_section = body.find("\n## ", idx + 1)
        if next_section == -1:
            next_section = len(body)
        return body[idx + len(f"## {section_title}"):next_section].strip()
