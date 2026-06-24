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

    # ========================================================================
    # M5 Alpha-GPT 5-subagent 编排 (基于 .agent/agents/alpha-gpt-*.md)
    # ========================================================================

    "alpha-gpt-idea-generator": AgentDefinition(
        id="alpha-gpt-idea-generator",
        name="Alpha-GPT Idea Generator",
        description=(
            "Alpha-GPT 第 1 阶段：根据 objective 生成 N 个 alpha 因子想法。"
            "无工具，纯文本生成。"
        ),
        mode="subagent",
        temperature=0.8,
        max_iterations=3,
        tools_denied={
            "sandbox", "bash", "shell", "edit", "write", "file_ops",
            "git_ops", "alpha_evaluate", "alpha_backtest",
            "backtest", "pipeline", "factor", "config_backtest",
        },
    ),

    "alpha-gpt-formula-translator": AgentDefinition(
        id="alpha-gpt-formula-translator",
        name="Alpha-GPT Formula Translator",
        description=(
            "Alpha-GPT 第 2 阶段：把 alpha 想法翻译成 polars 公式。"
            "无工具，纯文本生成 + 公式白名单校验。"
        ),
        mode="subagent",
        temperature=0.5,
        max_iterations=3,
        tools_denied={
            "sandbox", "bash", "shell", "edit", "write", "file_ops",
            "git_ops", "alpha_evaluate", "alpha_backtest",
            "backtest", "pipeline", "factor", "config_backtest",
        },
    ),

    "alpha-gpt-evaluator": AgentDefinition(
        id="alpha-gpt-evaluator",
        name="Alpha-GPT Evaluator",
        description=(
            "Alpha-GPT 第 3 阶段：调 alpha_evaluate / alpha_backtest 工具"
            "对公式做 IC/IR/Trading 回测评估。"
        ),
        mode="subagent",
        temperature=0.3,
        max_iterations=5,
        tools_allowed={
            "alpha_evaluate", "alpha_backtest", "read", "glob",
        },
    ),

    "alpha-gpt-reflector": AgentDefinition(
        id="alpha-gpt-reflector",
        name="Alpha-GPT Reflector",
        description=(
            "Alpha-GPT 第 4 阶段：基于评估结果反思 keep/mutate/drop，"
            "给下一轮 IdeaGenerator 改进建议。无工具。"
        ),
        mode="subagent",
        temperature=0.5,
        max_iterations=3,
        tools_denied={
            "sandbox", "bash", "shell", "edit", "write", "file_ops",
            "git_ops", "alpha_evaluate", "alpha_backtest",
            "backtest", "pipeline", "factor", "config_backtest",
        },
    ),

    "alpha-gpt-critic": AgentDefinition(
        id="alpha-gpt-critic",
        name="Alpha-GPT Critic",
        description=(
            "Alpha-GPT 第 5 阶段：从所有历史公式中选最终 top-K，"
            "综合 IR / 衰减 / mutual_IC 评分。无工具。"
        ),
        mode="subagent",
        temperature=0.2,
        max_iterations=3,
        tools_denied={
            "sandbox", "bash", "shell", "edit", "write", "file_ops",
            "git_ops", "alpha_evaluate", "alpha_backtest",
            "backtest", "pipeline", "factor", "config_backtest",
        },
    ),
}
