# QuantNodes Agent 安全增强设计文档

借鉴 OpenCode 设计模式，增强 QuantNodes Agent 的安全性、可控性和可扩展性。

**文档版本**: v1.0
**日期**: 2026-05-13
**状态**: 设计阶段

---

## 1. 背景与目标

### 1.1 现状问题

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| 无权限控制 | Agent 可执行任何 shell 命令 | 🔴 高 |
| 无 shell 超时 | 危险命令可能无限运行 | 🔴 高 |
| 无项目边界检查 | Agent 可读写项目外文件 | 🟡 中 |
| .env 文件无保护 | 敏感凭证可能被泄露 | 🔴 高 |
| 单 Agent 模式 | 无法区分只读分析和写入操作 | 🟡 中 |
| 上下文压缩粗糙 | 简单截断丢失重要信息 | 🟡 中 |

### 1.2 设计目标

1. **安全第一**：危险操作必须经过用户审批
2. **最小权限**：每个 Agent 只拥有完成任务所需的最小权限
3. **项目隔离**：限制文件系统访问范围在项目目录内
4. **可观测**：所有权限决策可审计
5. **渐进增强**：不破坏现有功能，逐步引入安全机制

---

## 2. 权限系统设计

### 2.1 核心概念

借鉴 OpenCode 的三动作模型：

```
操作 → 权限规则匹配 → allow（放行）/ deny（拒绝）/ ask（询问用户）
```

### 2.2 数据模型

```python
# QuantNodes/agent/permission/models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import fnmatch


class Action(Enum):
    """权限动作"""
    ALLOW = "allow"   # 自动放行
    DENY = "deny"     # 自动拒绝
    ASK = "ask"       # 询问用户


@dataclass(frozen=True)
class PermissionRule:
    """权限规则：(permission, pattern, action)"""
    permission: str      # 工具名称或类别，如 "bash", "edit", "read"
    pattern: str         # 目标模式，如 "*.py", "/path/to/file", "*"
    action: Action       # 动作

    def matches(self, permission: str, target: str) -> bool:
        """检查是否匹配"""
        perm_match = fnmatch.fnmatch(permission, self.permission)
        target_match = fnmatch.fnmatch(target, self.pattern)
        return perm_match and target_match


@dataclass
class PermissionRequest:
    """权限请求"""
    id: str                          # 请求 ID
    session_id: str                  # 会话 ID
    tool: str                        # 工具名称
    permission: str                  # 权限类别
    patterns: List[str]              # 目标模式列表
    always_patterns: List[str] = field(default_factory=list)  # 可永久允许的模式
    metadata: dict = field(default_factory=dict)               # 附加信息（如 diff）


@dataclass
class PermissionReply:
    """权限回复"""
    response: str  # "once" | "always" | "reject"
    message: str | None = None  # 拒绝时的反馈信息


# 类型别名
Ruleset = List[PermissionRule]
```

### 2.3 规则引擎

```python
# QuantNodes/agent/permission/evaluate.py

from typing import List
from .models import PermissionRule, Action


def evaluate(
    permission: str,
    target: str,
    *rulesets: Ruleset,
) -> PermissionRule:
    """评估权限规则

    规则评估逻辑：
    1. 将所有 ruleset 扁平化
    2. 从后向前查找第一个匹配的规则（后定义的规则优先）
    3. 如果没有匹配规则，默认返回 ask（安全默认）

    Args:
        permission: 权限类别（如 "bash", "edit"）
        target: 目标模式（如 "git commit", "/path/to/file.py"）
        *rulesets: 规则集（按优先级从低到高排列）

    Returns:
        匹配的规则（包含 action）
    """
    all_rules = []
    for rs in rulesets:
        all_rules.extend(rs)

    # 从后向前查找，后定义的规则优先
    for rule in reversed(all_rules):
        if rule.matches(permission, target):
            return rule

    # 默认：询问用户（安全默认）
    return PermissionRule(
        permission=permission,
        pattern="*",
        action=Action.ASK,
    )
```

### 2.4 权限服务

```python
# QuantNodes/agent/permission/service.py

import asyncio
from typing import Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field

from .models import (
    Action, PermissionRule, PermissionRequest, PermissionReply, Ruleset
)
from .evaluate import evaluate


@dataclass
class PendingRequest:
    """待处理的权限请求"""
    info: PermissionRequest
    deferred: asyncio.Future


class PermissionService:
    """权限管理服务

    职责：
    1. 管理权限规则集
    2. 处理工具执行前的权限检查
    3. 管理用户审批流程
    4. 持久化已批准的规则
    """

    def __init__(self, ruleset: Ruleset | None = None):
        self._ruleset = ruleset or []
        self._approved: Ruleset = []  # 已批准的规则（持久化）
        self._pending: Dict[str, PendingRequest] = {}
        self._reply_handler: Callable[[PermissionRequest], Awaitable[PermissionReply]] | None = None

    def set_reply_handler(self, handler: Callable[[PermissionRequest], Awaitable[PermissionReply]]):
        """设置回复处理器（用于 UI 集成）"""
        self._reply_handler = handler

    def add_rule(self, rule: PermissionRule) -> None:
        """添加规则"""
        self._ruleset.append(rule)

    def add_approved(self, rule: PermissionRule) -> None:
        """添加已批准的规则"""
        self._approved.append(rule)

    async def check(
        self,
        tool: str,
        permission: str,
        patterns: list[str],
        always_patterns: list[str] | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """检查权限

        Args:
            tool: 工具名称
            permission: 权限类别
            patterns: 目标模式列表
            always_patterns: 可永久允许的模式
            metadata: 附加信息

        Returns:
            True if allowed, False if denied

        Raises:
            PermissionDeniedError: 如果权限被拒绝
            PermissionRejectedError: 如果用户拒绝
        """
        always_patterns = always_patterns or []
        metadata = metadata or {}

        # 检查每个模式
        for pattern in patterns:
            rule = evaluate(
                permission,
                pattern,
                self._ruleset,
                self._approved,
            )

            if rule.action == Action.DENY:
                raise PermissionDeniedError(
                    f"Permission denied: {permission} on {pattern}"
                )

            if rule.action == Action.ALLOW:
                continue

            # Action.ASK: 需要用户审批
            if self._reply_handler is None:
                # 无回复处理器时，默认拒绝（安全默认）
                raise PermissionDeniedError(
                    f"No reply handler configured, permission denied: {permission} on {pattern}"
                )

            request = PermissionRequest(
                id=f"{tool}_{pattern}",
                session_id="",  # 由调用方设置
                tool=tool,
                permission=permission,
                patterns=patterns,
                always_patterns=always_patterns,
                metadata=metadata,
            )

            reply = await self._reply_handler(request)

            if reply.response == "reject":
                raise PermissionRejectedError(
                    reply.message or f"User rejected: {permission} on {pattern}"
                )

            if reply.response == "always":
                # 添加到已批准列表
                for p in always_patterns:
                    self._approved.append(PermissionRule(
                        permission=permission,
                        pattern=p,
                        action=Action.ALLOW,
                    ))

            # "once": 仅本次允许，继续

        return True

    def get_disabled_tools(self) -> set[str]:
        """获取被禁用的工具列表"""
        disabled = set()
        for rule in self._ruleset:
            if rule.pattern == "*" and rule.action == Action.DENY:
                disabled.add(rule.permission)
        return disabled


class PermissionDeniedError(Exception):
    """权限被拒绝"""
    pass


class PermissionRejectedError(Exception):
    """用户拒绝"""
    pass
```

### 2.5 默认规则集

```python
# QuantNodes/agent/permission/defaults.py

from pathlib import Path
from .models import PermissionRule, Action


def create_default_ruleset(project_root: str | Path) -> list[PermissionRule]:
    """创建默认权限规则集

    安全原则：
    1. 默认询问（安全默认）
    2. 项目内文件读取允许
    3. 敏感文件（.env）需要审批
    4. 外部目录访问需要审批
    5. shell 命令需要审批
    """
    project_root = Path(project_root)

    return [
        # === 基础规则 ===
        # 默认询问所有操作
        PermissionRule("*", "*", Action.ASK),

        # === 读取规则 ===
        # 项目内文件读取允许
        PermissionRule("read", "*", Action.ALLOW),
        # .env 文件需要审批
        PermissionRule("read", "*.env", Action.ASK),
        PermissionRule("read", "*.env.*", Action.ASK),
        PermissionRule("read", ".env", Action.ASK),
        PermissionRule("read", ".env.*", Action.ASK),
        # .env.example 是安全的
        PermissionRule("read", "*.env.example", Action.ALLOW),

        # === 编辑规则 ===
        # 项目内文件编辑允许（会通过 diff 展示给用户）
        PermissionRule("edit", "*", Action.ALLOW),

        # === Shell 规则 ===
        # 所有 shell 命令需要审批
        PermissionRule("bash", "*", Action.ASK),

        # === 外部目录规则 ===
        # 外部目录访问需要审批
        PermissionRule("external_directory", "*", Action.ASK),
        # /tmp 目录允许
        PermissionRule("external_directory", "/tmp/*", Action.ALLOW),

        # === Web 规则 ===
        # Web 抓取允许
        PermissionRule("webfetch", "*", Action.ALLOW),
        # Web 搜索允许
        PermissionRule("websearch", "*", Action.ALLOW),
    ]
```

---

## 3. Shell 安全设计

### 3.1 命令超时

```python
# QuantNodes/agent/tools/shell_safety.py

import asyncio
import signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShellConfig:
    """Shell 执行配置"""
    timeout_seconds: int = 120          # 默认 2 分钟超时
    max_output_bytes: int = 1024 * 1024  # 最大输出 1MB
    max_output_lines: int = 10000        # 最大输出行数
    project_root: str | None = None      # 项目根目录


async def execute_with_timeout(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
    env: dict | None = None,
) -> tuple[int, str, str]:
    """执行 shell 命令（带超时）

    Args:
        command: 要执行的命令
        cwd: 工作目录
        timeout: 超时秒数
        env: 环境变量

    Returns:
        (exit_code, stdout, stderr)
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        return proc.returncode or 0, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ShellTimeoutError(
            f"Command timed out after {timeout}s: {command[:100]}"
        )


class ShellTimeoutError(Exception):
    """Shell 命令超时"""
    pass
```

### 3.2 项目边界检查

```python
# QuantNodes/agent/tools/path_check.py

from pathlib import Path


def is_within_project(filepath: str, project_root: str) -> bool:
    """检查文件路径是否在项目目录内

    Args:
        filepath: 文件路径
        project_root: 项目根目录

    Returns:
        True if within project, False otherwise
    """
    try:
        file_path = Path(filepath).resolve()
        project_path = Path(project_root).resolve()
        return file_path.is_relative_to(project_path)
    except (ValueError, OSError):
        return False


def assert_within_project(filepath: str, project_root: str) -> None:
    """断言文件路径在项目目录内

    Raises:
        ExternalDirectoryError: 如果文件在项目外
    """
    if not is_within_project(filepath, project_root):
        raise ExternalDirectoryError(
            f"Access to external directory denied: {filepath}"
        )


class ExternalDirectoryError(Exception):
    """外部目录访问错误"""
    pass
```

### 3.3 输出截断

```python
# QuantNodes/agent/tools/output_truncation.py

from dataclasses import dataclass


@dataclass
class TruncatedOutput:
    """截断后的输出"""
    content: str
    truncated: bool
    total_lines: int
    total_bytes: int
    kept_lines: int
    kept_bytes: int


def truncate_output(
    output: str,
    max_lines: int = 10000,
    max_bytes: int = 1024 * 1024,
) -> TruncatedOutput:
    """截断输出

    策略：
    1. 保留前 max_lines/2 行
    2. 保留后 max_lines/2 行
    3. 中间用省略标记替代
    4. 同时检查字节限制

    Args:
        output: 原始输出
        max_lines: 最大行数
        max_bytes: 最大字节数

    Returns:
        TruncatedOutput 对象
    """
    lines = output.split("\n")
    total_lines = len(lines)
    total_bytes = len(output.encode("utf-8"))

    truncated = False
    kept_lines = total_lines
    kept_bytes = total_bytes

    if total_lines > max_lines or total_bytes > max_bytes:
        truncated = True
        half = max_lines // 2
        kept_lines = min(total_lines, max_lines)

        if total_lines > max_lines:
            kept_lines = max_lines
            head = lines[:half]
            tail = lines[-(max_lines - half):]
            lines = head + ["... (truncated) ..."] + tail

        result = "\n".join(lines)

        # 再检查字节限制
        if len(result.encode("utf-8")) > max_bytes:
            result = result[:max_bytes] + "\n... (truncated by bytes) ..."
            kept_bytes = max_bytes
    else:
        result = output

    return TruncatedOutput(
        content=result,
        truncated=truncated,
        total_lines=total_lines,
        total_bytes=total_bytes,
        kept_lines=kept_lines,
        kept_bytes=kept_bytes,
    )
```

---

## 4. Agent 多模式设计

### 4.1 Agent 定义

```python
# QuantNodes/agent/agents/definition.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ..permission.models import PermissionRule, Action


@dataclass
class AgentDefinition:
    """Agent 定义"""
    id: str                          # 唯一标识
    name: str                        # 显示名称
    description: str                 # 描述
    mode: str                        # "primary" | "subagent"
    permission_rules: list[PermissionRule] = field(default_factory=list)
    model: str | None = None         # 覆盖默认模型
    max_iterations: int = 20         # 最大迭代次数
    temperature: float = 0.7         # 温度
    tools_allowed: set[str] | None = None  # 允许的工具（None=全部）
    tools_denied: set[str] = field(default_factory=set)  # 禁止的工具
    system_prompt_override: str | None = None  # 覆盖系统提示词


# === 预定义 Agent ===

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
            # 只允许只读工具
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
```

### 4.2 Agent 切换机制

```python
# QuantNodes/agent/agents/manager.py

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

    def get_permission_service(self, agent_id: str | None = None) -> PermissionService:
        """获取 Agent 的权限服务

        每个 Agent 有独立的权限服务，规则集根据 Agent 定义合并。
        """
        agent_id = agent_id or self._current_agent

        if agent_id not in self._permission_services:
            # 合并默认规则和 Agent 特定规则
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
```

---

## 5. 工具框架增强

### 5.1 工具上下文

```python
# QuantNodes/agent/tools/context.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Awaitable
import asyncio

from ..permission.service import PermissionService, PermissionDeniedError


@dataclass
class ToolContext:
    """工具执行上下文

    传递给每个工具的执行上下文，包含权限检查、会话信息等。
    """
    session_id: str
    agent_id: str
    project_root: str
    permission_service: PermissionService
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)

    async def check_permission(
        self,
        tool: str,
        permission: str,
        patterns: list[str],
        always_patterns: list[str] | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """检查权限

        工具在执行危险操作前调用此方法。

        Raises:
            PermissionDeniedError: 如果权限被拒绝
            PermissionRejectedError: 如果用户拒绝
        """
        return await self.permission_service.check(
            tool=tool,
            permission=permission,
            patterns=patterns,
            always_patterns=always_patterns or [],
            metadata=metadata or {},
        )
```

### 5.2 增强的工具基类

```python
# QuantNodes/agent/tools/base_v2.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolCategory(Enum):
    """工具类别"""
    READ = "read"          # 只读操作
    WRITE = "write"        # 写入操作
    EXECUTE = "execute"    # 执行操作
    WEB = "web"            # Web 操作
    ANALYSIS = "analysis"  # 分析操作


@dataclass
class ToolMetadata:
    """工具元数据"""
    category: ToolCategory
    permission_key: str          # 权限键（用于权限检查）
    requires_approval: bool      # 是否需要用户审批
    timeout_seconds: int | None  # 执行超时（None=不超时）
    max_output_chars: int = 4000 # 最大输出字符数


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    content: Any
    error: str | None = None
    metadata: dict | None = None  # 附加元数据（如截断信息）


class ToolV2(ABC):
    """工具基类 V2

    增强功能：
    1. 类别和元数据
    2. 权限键定义
    3. 输出截断
    4. 超时控制
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """工具元数据"""
        pass

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

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        pass
```

---

## 6. 集成方案

### 6.1 与现有代码的集成点

```
QuantNodes/agent/
├── permission/           # 新增：权限系统
│   ├── __init__.py
│   ├── models.py         # 数据模型
│   ├── evaluate.py       # 规则引擎
│   ├── service.py        # 权限服务
│   └── defaults.py       # 默认规则集
│
├── agents/               # 新增：Agent 定义
│   ├── __init__.py
│   ├── definition.py     # Agent 定义
│   └── manager.py        # Agent 管理器
│
├── tools/
│   ├── base.py           # 修改：添加 metadata 属性
│   ├── registry.py       # 修改：集成权限检查
│   ├── shell_safety.py   # 新增：Shell 安全
│   ├── path_check.py     # 新增：路径检查
│   └── output_truncation.py  # 新增：输出截断
│
├── core/
│   ├── loop.py           # 修改：集成 Agent 切换
│   └── runner.py         # 修改：集成权限检查
│
└── config/
    └── settings.json     # 修改：添加权限配置
```

### 6.2 与现有 ToolRegistry 的集成

```python
# QuantNodes/agent/tools/registry.py (修改)

class ToolRegistry:
    def __init__(self, permission_service: PermissionService | None = None):
        self._tools: Dict[str, Tool] = {}
        self._permission_service = permission_service

    async def execute_tool(self, name: str, **kwargs: Any) -> ToolExecutionResult:
        """执行工具（带权限检查）"""
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                content=None,
                error=f"Tool '{name}' not found"
            )

        # 权限检查
        if self._permission_service and hasattr(tool, 'metadata'):
            try:
                await self._permission_service.check(
                    tool=name,
                    permission=tool.metadata.permission_key,
                    patterns=self._extract_patterns(tool, kwargs),
                    always_patterns=tool.metadata.always_patterns if hasattr(tool.metadata, 'always_patterns') else [],
                )
            except (PermissionDeniedError, PermissionRejectedError) as e:
                return ToolExecutionResult(
                    tool_name=name,
                    success=False,
                    content=None,
                    error=f"Permission denied: {e}"
                )

        # 执行工具
        try:
            params = tool.cast_params(kwargs)
            errors = tool.validate_params(params)
            if errors:
                return ToolExecutionResult(
                    tool_name=name,
                    success=False,
                    content=None,
                    error=f"Parameter validation failed: {', '.join(errors)}"
                )

            result = await tool.execute(**params)
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                content=result
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                content=None,
                error=str(e)
            )

    def _extract_patterns(self, tool: Tool, kwargs: dict) -> list[str]:
        """从工具参数中提取目标模式"""
        # 根据工具类型提取不同的模式
        if tool.name in ("read", "write", "edit"):
            return [kwargs.get("filepath", kwargs.get("file", "*"))]
        if tool.name == "sandbox":
            return ["*"]  # sandbox 始终需要审批
        return ["*"]
```

---

## 7. 配置格式

### 7.1 settings.json 扩展

```json
{
  "agent": {
    "default_agent": "build",
    "permission": {
      "enabled": true,
      "default_action": "ask",
      "rules": {
        "read": {
          "*": "allow",
          "*.env": "ask",
          "*.env.*": "ask"
        },
        "edit": {
          "*": "allow"
        },
        "bash": {
          "*": "ask"
        },
        "external_directory": {
          "*": "ask",
          "/tmp/*": "allow"
        }
      },
      "shell": {
        "timeout_seconds": 120,
        "max_output_bytes": 1048576
      }
    },
    "agents": {
      "build": {
        "model": "",
        "max_iterations": 20
      },
      "plan": {
        "model": "",
        "max_iterations": 10,
        "tools_denied": ["sandbox", "git_ops", "file_ops"]
      }
    }
  }
}
```

---

## 8. 实施计划

### Phase 1：权限系统核心（2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现 PermissionRule, Action 数据模型 | `permission/models.py` | P0 |
| 实现 evaluate() 规则引擎 | `permission/evaluate.py` | P0 |
| 实现 PermissionService | `permission/service.py` | P0 |
| 实现默认规则集 | `permission/defaults.py` | P0 |
| 编写单元测试 | `tests/test_permission.py` | P0 |

### Phase 2：Shell 安全（1 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现命令超时 | `tools/shell_safety.py` | P0 |
| 实现项目边界检查 | `tools/path_check.py` | P0 |
| 实现输出截断 | `tools/output_truncation.py` | P1 |
| 集成到 SandboxTool | `tools/sandbox.py` | P0 |

### Phase 3：Agent 多模式（2 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现 AgentDefinition | `agents/definition.py` | P0 |
| 实现 AgentManager | `agents/manager.py` | P0 |
| 集成到 AgentLoop | `core/loop.py` | P1 |
| 添加模式切换 API | `api/routers/agent.py` | P1 |

### Phase 4：工具框架增强（1 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现 ToolContext | `tools/context.py` | P1 |
| 实现 ToolV2 基类 | `tools/base_v2.py` | P1 |
| 集成到 ToolRegistry | `tools/registry.py` | P1 |
| 迁移现有工具 | 各工具文件 | P2 |

### Phase 5：上下文压缩（1 天）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现智能压缩（调用 LLM） | `core/compaction.py` | P1 |
| 替换简单截断 | `core/loop.py` | P1 |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 权限系统过于频繁询问 | 用户体验下降 | 提供 "always allow" 选项，持久化已批准规则 |
| Shell 超时误杀长时间任务 | 任务失败 | 允许工具自定义超时，提供取消超时选项 |
| Agent 切换导致状态丢失 | 上下文丢失 | 保持会话历史，切换时注入 Agent 特定上下文 |
| 现有工具不兼容 V2 基类 | 迁移成本高 | 保持 ToolV1 兼容，渐进迁移 |

---

## 10. 参考资料

- OpenCode 权限系统：`packages/opencode/src/permission/`
- OpenCode Agent 定义：`packages/opencode/src/agent/agent.ts`
- OpenCode Shell 工具：`packages/opencode/src/tool/shell.ts`
- OpenCode 快照系统：`packages/opencode/src/snapshot/`

---

## 11. 详细执行计划

### 11.1 总览

两条并行路线，先修 LLM 让系统能跑，再加安全增强。

```
Day 1:   路线 A - 修复 LLM 调用（让 Chat 能用）
Day 2-8: 路线 B - Agent 安全增强（本文档覆盖）
```

### 11.2 路线 A：LLM 调用层修复（Day 1）

#### 执行结果

**2026-05-14 更新**：路线 A 已完成，但因 OpenRouter API 服务端故障（500 Internal Server Error），切换为 MiniMax API。

#### 原计划（OpenRouter）

1. 将 `model` 改为 `google/gemma-4-31b-it:free`
2. 修复 URL 清理逻辑（不再 strip `/v1`）
3. 添加 LiteLLM prefix stripping

**问题**：OpenRouter API 返回 500，无法验证

#### 最终方案（MiniMax）

**修改文件**: `.quant_agent/settings.json`

```json
{
  "agent": {
    "provider": "openai",
    "model": "MiniMax-M2.7",
    "api_key": "sk-cp-ZsXN5_...",  // MiniMax API Key
    "api_base": "https://api.minimaxi.com/v1",
    "use_litellm": false,
    "llm_timeout": 60
  }
}
```

**验证**:
- ✅ OpenAIClient 直接调用成功
- ✅ QuantNodesLLMProvider.chat() 测试成功
- ✅ 45 个 provider 测试通过

**注意**: LiteLLM 不支持 MiniMax（需要使用 `use_litellm: false`）

#### Step A4：端到端测试

```bash
# 重启服务
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 19380

# 测试 chat 端点
curl -X POST http://localhost:19380/api/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "hello", "session_id": "test"}'

# 预期: 返回 200 + 正常响应
```

### 11.3 路线 B：Agent 安全增强（Day 2-8）

#### Phase 1：权限系统核心（Day 2-3）

**Step B1：创建权限模块目录结构**

新建目录: `QuantNodes/agent/permission/`

新建文件:
```
QuantNodes/agent/permission/
├── __init__.py
├── models.py
├── evaluate.py
├── service.py
└── defaults.py
```

**Step B2：实现 `models.py`**

文件: `QuantNodes/agent/permission/models.py`

内容:
- `Action(Enum)`: ALLOW, DENY, ASK
- `PermissionRule`: (permission, pattern, action) + matches() 方法
- `PermissionRequest`: 工具执行前的权限请求
- `PermissionReply`: 用户回复（once/always/reject）
- `PermissionDeniedError`: 权限拒绝异常
- `PermissionRejectedError`: 用户拒绝异常

测试: `tests/agent/test_permission_models.py`
- 测试 `PermissionRule.matches()` 的各种匹配场景
- 测试 `Action` 枚举值

**Step B3：实现 `evaluate.py`**

文件: `QuantNodes/agent/permission/evaluate.py`

内容:
- `evaluate(permission, target, *rulesets) -> PermissionRule`
- 逻辑: 扁平化所有规则集，从后向前查找，第一个匹配的规则胜出
- 默认: 无匹配规则时返回 ASK（安全默认）

测试: `tests/agent/test_permission_evaluate.py`
- 测试规则优先级（后定义的规则优先）
- 测试通配符匹配
- 测试无匹配时的默认行为

**Step B4：实现 `service.py`**

文件: `QuantNodes/agent/permission/service.py`

内容:
```python
class PermissionService:
    - __init__(ruleset, approved_rules)
    - set_reply_handler(handler)  # 设置 UI 回调
    - add_rule(rule)              # 添加规则
    - add_approved(rule)          # 添加已批准规则
    - check(tool, permission, patterns, always_patterns, metadata)  # 权限检查
    - get_disabled_tools()        # 获取禁用工具列表
```

测试: `tests/agent/test_permission_service.py`
- 测试 allow/deny/ask 三种动作
- 测试 "always" 批准后的持久化
- 测试无 reply_handler 时的默认拒绝

**Step B5：实现 `defaults.py`**

文件: `QuantNodes/agent/permission/defaults.py`

内容:
```python
def create_default_ruleset(project_root) -> list[PermissionRule]:
    - 默认 ASK 所有操作
    - read: 项目内 ALLOW，.env ASK
    - edit: 项目内 ALLOW
    - bash: ASK
    - external_directory: ASK（/tmp ALLOW）
    - webfetch/websearch: ALLOW
```

测试: `tests/agent/test_permission_defaults.py`

#### Phase 2：Shell 安全（Day 4）

**Step B6：实现 `shell_safety.py`**

文件: `QuantNodes/agent/tools/shell_safety.py`

内容:
- `ShellConfig`: timeout_seconds=120, max_output_bytes=1MB, max_output_lines=10000
- `execute_with_timeout(command, cwd, timeout, env)`: 执行命令带超时
- `ShellTimeoutError`: 超时异常

测试: `tests/agent/test_shell_safety.py`

**Step B7：实现 `path_check.py`**

文件: `QuantNodes/agent/tools/path_check.py`

内容:
- `is_within_project(filepath, project_root) -> bool`
- `assert_within_project(filepath, project_root)`: raises ExternalDirectoryError
- `ExternalDirectoryError`: 外部目录访问异常

测试: `tests/agent/test_path_check.py`

**Step B8：实现 `output_truncation.py`**

文件: `QuantNodes/agent/tools/output_truncation.py`

内容:
- `TruncatedOutput`: dataclass with content, truncated, stats
- `truncate_output(output, max_lines, max_bytes)`: 截断输出

测试: `tests/agent/test_output_truncation.py`

**Step B9：集成到 SandboxTool**

文件: `QuantNodes/agent/tools/sandbox.py`

修改:
- 添加 `ToolContext` 参数到 `execute()`
- 在执行前调用 `ctx.check_permission()`
- 添加超时控制

#### Phase 3：Agent 多模式（Day 5-6）

**Step B10：创建 Agent 定义模块**

新建目录: `QuantNodes/agent/agents/`

新建文件:
```
QuantNodes/agent/agents/
├── __init__.py
├── definition.py
└── manager.py
```

**Step B11：实现 `definition.py`**

文件: `QuantNodes/agent/agents/definition.py`

内容:
- `AgentDefinition`: dataclass with id, name, mode, permission_rules, tools_allowed/denied
- `QUANTNODES_AGENTS`: 预定义 4 个 Agent（build/plan/explore/backtest）

测试: `tests/agent/test_agent_definition.py`

**Step B12：实现 `manager.py`**

文件: `QuantNodes/agent/agents/manager.py`

内容:
```python
class AgentManager:
    - get_current_agent() -> AgentDefinition
    - set_agent(agent_id) -> AgentDefinition
    - get_permission_service(agent_id) -> PermissionService
    - list_agents() -> list[AgentDefinition]
    - register_agent(agent)
```

测试: `tests/agent/test_agent_manager.py`

**Step B13：集成到 AgentLoop**

文件: `QuantNodes/agent/core/loop.py`

修改:
- `__init__`: 创建 `AgentManager` 实例
- `chat_stream`: 在工具执行前调用权限检查
- 添加 `switch_agent(agent_id)` 方法
- 添加 `list_agents()` 方法

**Step B14：添加 API 端点**

文件: `api/routers/agent.py`

新增端点:
```
GET  /api/agent/agents        # 列出所有 Agent
POST /api/agent/switch        # 切换 Agent
GET  /api/agent/current       # 获取当前 Agent
```

#### Phase 4：工具框架增强（Day 7）

**Step B15：实现 `context.py`**

文件: `QuantNodes/agent/tools/context.py`

内容:
```python
@dataclass
class ToolContext:
    session_id: str
    agent_id: str
    project_root: str
    permission_service: PermissionService

    async def check_permission(tool, permission, patterns, ...)
```

**Step B16：增强 `registry.py`**

文件: `QuantNodes/agent/tools/registry.py`

修改:
- `__init__`: 接受 `PermissionService` 参数
- `execute_tool`: 执行前调用权限检查

**Step B17：迁移现有工具（渐进式）**

| 优先级 | 工具 | 修改内容 |
|--------|------|----------|
| P0 | `sandbox.py` | 添加 `ToolContext` 参数 |
| P0 | `file_ops.py` | 添加路径检查 + 权限检查 |
| P0 | `git_ops.py` | 添加 `git add -A` 警告 |
| P1 | `backtest.py` | 添加执行前确认 |
| P1 | `config_backtest.py` | 添加 SQL 注入防护 |

#### Phase 5：上下文压缩（Day 8）

**Step B18：实现智能压缩**

文件: `QuantNodes/agent/core/compaction.py`

内容:
```python
class ContextCompactor:
    - __init__(provider, config)
    - async compact(messages, target_tokens) -> list[dict]
    - 逻辑:
      1. 保留系统消息
      2. 保留最近 N 条消息
      3. 对中间消息调用 LLM 生成摘要
      4. 用摘要替代原始消息
```

测试: `tests/agent/test_compaction.py`

**Step B19：集成到 AgentLoop**

文件: `QuantNodes/agent/core/loop.py`

修改:
- 替换现有的 `_auto_compact` 方法
- 使用新的 `ContextCompactor`

---

## 12. 依赖关系图

```
Day 1: Step A1 → A2 → A3 → A4（串行）
       ↓
Day 2: Step B1 → B2 → B3 → B4 → B5（串行）
       ↓
Day 4: Step B6 → B7 → B8 → B9（串行）
       ↓
Day 5: Step B10 → B11 → B12（串行）
Day 6: Step B13 → B14（串行，依赖 B12）
       ↓
Day 7: Step B15 → B16 → B17（串行，依赖 B4）
       ↓
Day 8: Step B18 → B19（串行）
```

---

## 13. 测试策略

### 13.1 每个 Phase 的测试要求

| Phase | 测试文件 | 测试数量 | 通过标准 |
|-------|----------|----------|----------|
| B1 | `test_permission_*.py` | ~15 个 | 100% 通过 |
| B2 | `test_shell_safety.py` `test_path_check.py` `test_output_truncation.py` | ~10 个 | 100% 通过 |
| B3 | `test_agent_definition.py` `test_agent_manager.py` | ~8 个 | 100% 通过 |
| B4 | 现有工具测试 | ~45 个 | 100% 通过 |
| B5 | `test_compaction.py` | ~5 个 | 100% 通过 |

### 13.2 回归测试

每个 Phase 完成后运行:
```bash
python -m pytest tests/agent/ -v
```

确保现有测试不被破坏。

---

## 14. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| LiteLLM URL 修复引入新 bug | 中 | 高 | 先写单元测试，再修改代码 |
| 权限系统过于频繁询问 | 高 | 中 | 提供 "always allow" 选项 |
| 现有工具迁移破坏兼容性 | 中 | 高 | 渐进式迁移，保持 ToolV1 兼容 |
| 测试覆盖不足 | 中 | 中 | 每个 Step 都写测试 |
| 上下文压缩调用 LLM 增加成本 | 低 | 低 | 仅在必要时触发 |

---

## 15. 完成标准

### Day 1 完成标准
- [x] Chat 端点返回 200 + 正常响应（切换到 MiniMax 后验证成功）
- [x] 45 个 provider 测试通过
- [x] 可以和 Agent 进行简单对话（MiniMax-M2.7 可用）

### Day 8 完成标准
- [x] 所有新增测试通过（~40 个）- 2026-05-14
- [x] 所有现有测试通过（~783 个）- 2026-05-14
- [x] 权限系统可配置 - Phase 1 完成
- [x] Agent 多模式可切换 - Phase 3 完成
- [x] Shell 命令有超时保护 - Phase 2 完成
- [x] .env 文件有权限保护 - Phase 1 完成
- [x] 上下文压缩使用 LLM 智能摘要 - Phase 5 完成

**实现状态**:
- Phase 1 (Permission): ✅ `permission/models.py`, `evaluate.py`, `service.py`, `defaults.py`
- Phase 2 (Shell Safety): ✅ `tools/shell_safety.py`, `path_check.py`, `output_truncation.py`
- Phase 3 (Agent Multi-mode): ✅ `agents/definition.py`, `manager.py`
- Phase 4 (Tool Framework): ✅ `tools/context.py`, `tools/base_v2.py`
- Phase 5 (Compaction): ✅ `core/compaction.py`

---

## 附录：MiniMax Provider 实现计划

### 背景

LiteLLM SDK 不支持 MiniMax，需要创建独立的 MiniMax Provider。

### 目标

创建 `QuantNodes/agent/providers/minimax.py`，支持 MiniMax 特定功能。

### 实现内容

1. **MiniMaxProvider 类**（继承 LLMProvider）
2. **API 错误处理**：MiniMax 特定错误码处理
3. **专有参数支持**：MiniMax API 特有参数
4. **更新导出**：`providers/__init__.py`

### 文件结构

```
QuantNodes/agent/providers/
├── minimax.py      # 新建：MiniMax Provider
└── __init__.py    # 修改：添加 MiniMaxProvider 导出
```

### 优先级

低（当前 OpenAI 客户端 + `use_litellm: false` 已满足需求）
