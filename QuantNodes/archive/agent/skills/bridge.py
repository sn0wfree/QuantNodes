# coding=utf-8
"""
Skill → Tool Bridge

将 Skill 自动转换为 Agent Tool 并注册到 ToolRegistry。
"""

import logging
from typing import Any, Dict

from ..skills.base import Skill, SkillResult
from ..skills.registry import SkillRegistry
from ..tools.base import Tool
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillToolAdapter(Tool):
    """将 Skill 适配为 Tool 接口"""

    def __init__(self, skill: Skill):
        self._skill = skill

    @property
    def name(self) -> str:
        return f"skill_{self._skill.metadata.name}"

    @property
    def description(self) -> str:
        return self._skill.metadata.description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._skill.get_parameters_schema()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> Any:
        try:
            result: SkillResult = await self._skill.execute(kwargs)
            return result.to_dict()
        except Exception as e:
            logger.error("Skill %s execution failed: %s", self._skill.name, e)
            return {"success": False, "error": str(e), "data": None}


class SkillToolBridge:
    """将 SkillRegistry 中的 Skill 注册为 Agent Tool"""

    def __init__(self, skill_registry: SkillRegistry, tool_registry: ToolRegistry):
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry

    def register_all(self) -> int:
        """将所有 Skill 注册为 Tool，返回注册数量"""
        count = 0
        for skill in self.skill_registry.list_all():
            tool = SkillToolAdapter(skill)
            self.tool_registry.register(tool)
            count += 1
        return count

    def unregister_all(self) -> int:
        """移除所有 SkillToolAdapter"""
        count = 0
        for tool in list(self.tool_registry.list_tools()):
            if isinstance(tool, SkillToolAdapter):
                self.tool_registry.unregister(tool.name)
                count += 1
        return count
