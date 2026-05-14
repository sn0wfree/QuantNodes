# coding=utf-8
"""
Agent 定义模块
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from ..permission.models import PermissionRule, Action


@dataclass
class AgentDefinition:
    """Agent 定义"""
    id: str
    name: str
    description: str
    mode: str = "primary"
    permission_rules: list[PermissionRule] = field(default_factory=list)
    model: Optional[str] = None
    max_iterations: int = 20
    temperature: float = 0.7
    tools_allowed: Optional[set[str]] = None
    tools_denied: set[str] = field(default_factory=set)
    system_prompt_override: Optional[str] = None


QUANTNODES_AGENTS: Dict[str, AgentDefinition] = {
    "build": AgentDefinition(
        id="build",
        name="Build Agent",
        description="默认 Agent，具有完整权限，可执行代码修改和分析",
        mode="primary",
        permission_rules=[
            PermissionRule("*", "*", Action.ALLOW),
            PermissionRule("bash", "*", Action.ASK),
            PermissionRule("external_directory", "*", Action.ASK),
            PermissionRule("read", "*.env", Action.ASK),
            PermissionRule("read", "*.env.*", Action.ASK),
        ],
    ),

    "plan": AgentDefinition(
        id="plan",
        name="Plan Agent",
        description="只读分析 Agent，禁止文件修改，用于代码探索和策略分析",
        mode="primary",
        permission_rules=[
            PermissionRule("*", "*", Action.ALLOW),
            PermissionRule("edit", "*", Action.DENY),
            PermissionRule("bash", "*", Action.DENY),
            PermissionRule("write", "*", Action.DENY),
            PermissionRule("read", "*", Action.ALLOW),
            PermissionRule("glob", "*", Action.ALLOW),
            PermissionRule("grep", "*", Action.ALLOW),
            PermissionRule("code_search", "*", Action.ALLOW),
        ],
        tools_denied={"sandbox", "git_ops", "file_ops", "strategy", "backtest"},
    ),

    "explore": AgentDefinition(
        id="explore",
        name="Explore Agent",
        description="快速代码搜索子 Agent，只读，用于代码库探索",
        mode="subagent",
        permission_rules=[
            PermissionRule("*", "*", Action.DENY),
            PermissionRule("read", "*", Action.ALLOW),
            PermissionRule("glob", "*", Action.ALLOW),
            PermissionRule("grep", "*", Action.ALLOW),
            PermissionRule("code_search", "*", Action.ALLOW),
            PermissionRule("webfetch", "*", Action.ALLOW),
        ],
        tools_denied={
            "sandbox", "git_ops", "file_ops", "strategy", "backtest",
            "pipeline", "factor", "config_backtest", "wiki",
        },
    ),

    "backtest": AgentDefinition(
        id="backtest",
        name="Backtest Agent",
        description="回测专用 Agent，专注于策略回测和分析",
        mode="subagent",
        permission_rules=[
            PermissionRule("*", "*", Action.ALLOW),
            PermissionRule("bash", "*", Action.ASK),
        ],
        tools_allowed={
            "backtest", "config_backtest", "factor", "strategy",
            "read", "glob", "grep", "code_search", "wiki",
        },
    ),
}