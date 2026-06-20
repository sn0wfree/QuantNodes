# coding=utf-8
"""
Skill Loader - Progressive Skill Loader

Phase 4.1: Skill Infrastructure
"""

import importlib.util
from pathlib import Path
from typing import Dict, List, Any

from .base import Skill, SkillCategory
from .registry import SkillRegistry


class SkillLoader:
    """Progressive Skill Loader"""

    def __init__(self, registry: SkillRegistry = None):
        self.registry = registry or SkillRegistry()
        self._loaded: Dict[str, bool] = {}
        self._load_order: List[str] = []
        self._failed: Dict[str, str] = {}

    def discover_skills(self, skills_dir: Path) -> List[str]:
        """Discover all skills in the skills directory"""
        discovered = []
        if not skills_dir.exists():
            return discovered
        for item in skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                skill_file = item / f"{item.name}.py"
                if skill_file.exists():
                    discovered.append(str(item))
        return discovered

    async def load_skill(
        self, skill_path: Path, category: SkillCategory = None
    ) -> bool:
        """Load a single skill from path"""
        spec = importlib.util.spec_from_file_location("skill_module", skill_path)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            self._failed[str(skill_path)] = str(e)
            return False
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Skill) and attr != Skill:
                skill_instance = attr()
                if category:
                    skill_instance._metadata.category = category
                self.registry.register(skill_instance)
                self._loaded[skill_instance.name] = True
                self._load_order.append(skill_instance.name)
                return True
        return False

    async def load_all(self, skills_dir: Path) -> Dict[str, bool]:
        """Load all discovered skills"""
        results = {}
        for skill_path in self.discover_skills(skills_dir):
            skill_name = Path(skill_path).name
            success = await self.load_skill(
                Path(skill_path) / f"{skill_name}.py"
            )
            results[skill_name] = success
        return results

    async def progressive_load(
        self, skills_dir: Path, requested_skills: List[str]
    ) -> Dict[str, Skill]:
        """Progressive load: only load requested skills"""
        loaded = {}
        for skill_name in requested_skills:
            skill_path = skills_dir / skill_name / f"{skill_name}.py"
            if skill_path.exists() and skill_name not in self._loaded:
                success = await self.load_skill(skill_path)
                if success:
                    skill = self.registry.get(skill_name)
                    if skill:
                        loaded[skill_name] = skill
        return loaded

    def get_loading_status(self) -> Dict[str, Any]:
        """Get loading status"""
        return {
            "loaded": self._loaded,
            "load_order": self._load_order,
            "failed": self._failed,
            "total_loaded": len(self._loaded),
        }

    def reload_skill(self, name: str) -> bool:
        """Reload a skill"""
        if name in self._loaded:
            self._loaded.pop(name)
            self._load_order.remove(name)
            self.registry.unregister(name)
        return True

    def is_loaded(self, name: str) -> bool:
        """Check if a skill is loaded"""
        return name in self._loaded
