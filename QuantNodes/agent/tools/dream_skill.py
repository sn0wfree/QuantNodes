# coding=utf-8
"""
DreamSkill - Agent Tool for Querying Dreams

Phase 4.2: Dream System
"""

from typing import Any, Dict

from ..skills.base import Skill, SkillCategory, SkillMetadata, SkillResult


class DreamSkill(Skill):
    """Skill for querying Dream system insights"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dream_insight",
            description="查询 Dream 系统生成的投资洞察",
            category=SkillCategory.DREAM,
            examples=[
                "查看最近的投资洞察",
                "有什么新的市场发现",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute dream insight query"""
        from ..core.memory import MemoryStore

        workspace = context.get("workspace", ".")
        dream_store = MemoryStore(workspace).get_dream_store()

        query = context.get("query", "recent")
        limit = context.get("limit", 5)

        if query == "recent":
            dreams = dream_store.get_recent_dreams(limit)
        else:
            dreams = dream_store.get_dreams_by_type(query, limit)

        data = [
            {
                "id": d.id,
                "type": d.type,
                "content": d.content,
                "insights": d.insights,
                "source": d.source,
                "confidence": d.confidence,
                "timestamp": d.timestamp,
            }
            for d in dreams
        ]

        return SkillResult(success=True, data=data)

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "查询类型: recent 或具体类型 "
                        "(wiki_insight/factor_insight/strategy_insight)"
                    ),
                    "default": "recent",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量",
                    "default": 5,
                },
            },
        }
