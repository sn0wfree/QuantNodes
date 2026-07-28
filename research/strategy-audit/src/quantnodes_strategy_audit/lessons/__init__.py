"""Lesson mapping access.

Loads detector ↔ L-NNN lesson mapping from mapping.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LESSONS_PATH = Path(__file__).parent / "mapping.json"


def load_lessons() -> dict[str, Any]:
    """Load lesson mapping from JSON file."""
    if not _LESSONS_PATH.exists():
        return {}
    return json.loads(_LESSONS_PATH.read_text(encoding="utf-8"))


LESSON_MAPPING: dict[str, Any] = load_lessons().get("lessons", {})


def get_lesson(lesson_id: str) -> dict[str, Any] | None:
    """Get lesson by L-NNN ID."""
    return LESSON_MAPPING.get(lesson_id)


def get_detector_lessons(detector_name: str) -> list[str]:
    """Get all L-NNN lesson IDs associated with a detector."""
    lessons = []
    for lesson_id, mapping in LESSON_MAPPING.items():
        detectors = mapping.get("detectors", [])
        if detector_name in detectors:
            lessons.append(lesson_id)
    return lessons
