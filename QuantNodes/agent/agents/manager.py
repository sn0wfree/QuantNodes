# coding=utf-8
"""
Agent 管理器
"""

from typing import Dict, Optional
from .definition import AgentDefinition, QUANTNODES_AGENTS
from ..permission.service import PermissionService
from ..permission.defaults import create_default_ruleset


class AgentManager:
    """Agent 管理器

    职责：
    1. 管理 Agent 定义
    2. 处理 Agent 切换
    3. 为每个 Agent 创建权限服务
    """

    def __init__(self, project_root: str):
        self._agents = dict(QUANTNODES_AGENTS)
        self._current_agent: str = "build"
        self._project_root = project_root
        self._permission_services: Dict[str, PermissionService] = {}

    def get_current_agent(self) -> AgentDefinition:
        """获取当前 Agent"""
        return self._agents[self._current_agent]

    def set_agent(self, agent_id: str) -> AgentDefinition:
        """切换 Agent

        Args:
            agent_id: Agent ID

        Returns:
            新的 Agent 定义

        Raises:
            ValueError: 如果 Agent 不存在
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent not found: {agent_id}")

        self._current_agent = agent_id
        return self._agents[agent_id]

    def get_permission_service(self, agent_id: Optional[str] = None) -> PermissionService:
        """获取 Agent 的权限服务

        每个 Agent 有独立的权限服务，规则集根据 Agent 定义合并。
        """
        agent_id = agent_id or self._current_agent

        if agent_id not in self._permission_services:
            default_rules = create_default_ruleset(self._project_root)
            agent_rules = self._agents[agent_id].permission_rules
            combined_rules = default_rules + agent_rules

            self._permission_services[agent_id] = PermissionService(
                ruleset=combined_rules,
            )

        return self._permission_services[agent_id]

    def list_agents(self) -> list[AgentDefinition]:
        """列出所有 Agent"""
        return list(self._agents.values())

    def register_agent(self, agent: AgentDefinition) -> None:
        """注册自定义 Agent"""
        self._agents[agent.id] = agent