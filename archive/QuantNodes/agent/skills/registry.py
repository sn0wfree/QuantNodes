# coding=utf-8
"""
Skill Registry - Singleton Registry for Skills (Thread-Safe)

Phase 4.1: Skill Infrastructure
"""

import threading
from typing import Dict, List, Optional
from collections import defaultdict

from .base import Skill, SkillCategory


class SkillRegistry:
    """Skill Registry (Singleton + Thread-Safe)"""

    _instance: Optional["SkillRegistry"] = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._skills: Dict[str, Skill] = {}
                cls._instance._categories: Dict[
                    SkillCategory, List[str]
                ] = defaultdict(list)
                cls._instance._aliases: Dict[str, str] = {}
            return cls._instance

    def register(self, skill: Skill, aliases: List[str] = None) -> None:
        """Register a skill"""
        name = skill.name
        if name in self._skills:
            raise ValueError(f"Skill '{name}' already registered")
        self._skills[name] = skill
        self._categories[skill.category].append(name)
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name or alias"""
        resolved = self._aliases.get(name, name)
        return self._skills.get(resolved)

    def list_by_category(self, category: SkillCategory) -> List[Skill]:
        """List all skills in a category"""
        return [self._skills[name] for name in self._categories.get(category, [])]

    def list_all(self) -> List[Skill]:
        """List all registered skills"""
        return list(self._skills.values())

    def list_names(self) -> List[str]:
        """List all skill names"""
        return list(self._skills.keys())

    def exists(self, name: str) -> bool:
        """Check if a skill exists"""
        return name in self._skills or name in self._aliases

    def unregister(self, name: str) -> bool:
        """Unregister a skill"""
        resolved = self._aliases.pop(name, None) or name
        if resolved not in self._skills:
            return False
        skill = self._skills.pop(resolved)
        if resolved in self._categories[skill.category]:
            self._categories[skill.category].remove(resolved)
        return True

    def clear(self) -> None:
        """Clear all skills"""
        self._skills.clear()
        self._categories.clear()
        self._aliases.clear()

    def get_skill_info(self, name: str) -> Optional[Dict]:
        """Get skill information"""
        skill = self.get(name)
        if not skill:
            return None
        return {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category.value,
            "version": skill.metadata.version,
            "tags": skill.metadata.tags,
        }

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search skills by name or description"""
        results = []
        query_lower = query.lower()
        for skill in self._skills.values():
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
            ):
                results.append(self.get_skill_info(skill.name))
                if len(results) >= limit:
                    break
        return results