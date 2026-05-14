# coding=utf-8
"""
Tool V2 - Enhanced tool base class with metadata and permission support
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import Tool


class ToolCategory(Enum):
    """Tool category for permission and execution classification"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    WEB = "web"
    ANALYSIS = "analysis"


@dataclass
class ToolMetadata:
    """Metadata for tool permission and execution control"""
    category: ToolCategory
    permission_key: str
    requires_approval: bool = False
    timeout_seconds: Optional[int] = None
    max_output_chars: int = 4000
    always_patterns: List[str] = field(default_factory=list)
    danger_level: str = "low"


@dataclass
class ToolV2(Tool):
    """Enhanced tool base class V2

    Adds:
    - Tool metadata for permission checking
    - Output truncation support
    - Timeout configuration
    """
    metadata: ToolMetadata = field(default_factory=ToolMetadata(
        category=ToolCategory.READ,
        permission_key="read",
    ))

    @property
    def read_only(self) -> bool:
        return self.metadata.category == ToolCategory.READ

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def extract_patterns(self, **kwargs: Any) -> List[str]:
        """Extract target patterns from tool arguments for permission checking.

        Override this method in tools that work with file paths.
        """
        return ["*"]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        pass


class ToolV1Compat(ToolV2):
    """Compatibility wrapper for ToolV1 tools.

    Wraps an existing Tool instance and provides V2 functionality.
    """

    def __init__(self, wrapped_tool: Tool):
        self._wrapped = wrapped_tool
        self._metadata = ToolMetadata(
            category=ToolCategory.READ if wrapped_tool.read_only else ToolCategory.WRITE,
            permission_key="unknown",
        )

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._wrapped.parameters

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: ToolMetadata):
        self._metadata = value

    @property
    def read_only(self) -> bool:
        return self._wrapped.read_only

    async def execute(self, **kwargs: Any) -> Any:
        return await self._wrapped.execute(**kwargs)