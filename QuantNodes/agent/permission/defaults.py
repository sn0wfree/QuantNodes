# coding=utf-8
"""
默认权限规则集
"""

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
        PermissionRule("*", "*", Action.ASK),
        PermissionRule("read", "*", Action.ALLOW),
        PermissionRule("read", "*.env", Action.ASK),
        PermissionRule("read", "*.env.*", Action.ASK),
        PermissionRule("read", ".env", Action.ASK),
        PermissionRule("read", ".env.*", Action.ASK),
        PermissionRule("read", "*.env.example", Action.ALLOW),
        PermissionRule("edit", "*", Action.ALLOW),
        PermissionRule("bash", "*", Action.ASK),
        PermissionRule("external_directory", "*", Action.ASK),
        PermissionRule("external_directory", "/tmp/*", Action.ALLOW),
        PermissionRule("webfetch", "*", Action.ALLOW),
        PermissionRule("websearch", "*", Action.ALLOW),
    ]