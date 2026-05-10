# coding=utf-8
"""
Skill API Router - 连接真实 SkillRegistry

提供技能列表、详情、执行、分类查询的 REST API。
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from QuantNodes.agent.skills.registry import SkillRegistry


router = APIRouter()


class ExecuteRequest(BaseModel):
    query: str = ""
    params: Dict[str, Any] = {}


@router.get("/")
async def list_skills():
    """列出所有已注册技能"""
    registry = SkillRegistry()
    skills = registry.list_all()
    return [
        {
            "name": s.name,
            "description": s.description,
            "category": s.category.value,
            "version": s.metadata.version,
            "tags": s.metadata.tags,
            "status": s.metadata.status.value,
        }
        for s in skills
    ]


@router.get("/categories/list")
async def list_categories():
    """按类别分组列出技能"""
    registry = SkillRegistry()
    categories: Dict[str, List[str]] = {}
    for skill in registry.list_all():
        cat = skill.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(skill.name)
    return categories


@router.get("/{skill_name}")
async def get_skill(skill_name: str):
    """获取单个技能详情"""
    registry = SkillRegistry()
    info = registry.get_skill_info(skill_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    skill = registry.get(skill_name)
    info["parameters"] = skill.get_parameters_schema() if skill else {}
    return info


@router.post("/{skill_name}/execute")
async def execute_skill(skill_name: str, request: ExecuteRequest = None):
    """执行技能"""
    registry = SkillRegistry()
    skill = registry.get(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    context = {}
    if request:
        context["query"] = request.query
        context.update(request.params)

    try:
        result = await skill.execute(context)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")
