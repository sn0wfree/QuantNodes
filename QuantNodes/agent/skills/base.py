# coding=utf-8
"""
Skill System Base Classes

Phase 4.1: Skill Infrastructure
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class SkillCategory(Enum):
    STRATEGY = "strategy"
    FACTOR = "factor"
    ANALYSIS = "analysis"
    DREAM = "dream"


class SkillStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class SkillMetadata:
    name: str
    description: str
    category: SkillCategory
    version: str = "1.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: SkillStatus = SkillStatus.ACTIVE
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class SkillResult:
    success: bool
    data: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


class Skill(ABC):
    """Skill Abstract Base Class"""

    def __init__(self):
        self._metadata: Optional[SkillMetadata] = None

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """Return skill metadata"""
        pass

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def category(self) -> SkillCategory:
        return self.metadata.category

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute the skill"""
        pass

    def validate_context(
        self, context: Dict[str, Any], required_keys: List[str]
    ) -> bool:
        """Validate if context contains required keys"""
        return all(k in context for k in required_keys)

    def get_example_prompts(self) -> List[str]:
        """Return example prompts"""
        return self.metadata.examples

    def to_tool_schema(self) -> Dict[str, Any]:
        """Convert to nanobot Tool Schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
            "category": self.category.value,
        }

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter Schema"""
        pass
