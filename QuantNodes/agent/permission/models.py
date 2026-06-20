# coding=utf-8
"""
权限系统数据模型
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import fnmatch


class Action(Enum):
    """权限动作"""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PermissionRule:
    """权限规则：(permission, pattern, action)"""
    permission: str
    pattern: str
    action: Action

    def matches(self, permission: str, target: str) -> bool:
        perm_match = fnmatch.fnmatch(permission, self.permission)
        target_match = fnmatch.fnmatch(target, self.pattern)
        return perm_match and target_match


@dataclass
class PermissionRequest:
    """权限请求"""
    id: str
    session_id: str
    tool: str
    permission: str
    patterns: List[str]
    always_patterns: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class PermissionReply:
    """权限回复"""
    response: str
    message: Optional[str] = None


Ruleset = List[PermissionRule]


class PermissionDeniedError(Exception):
    pass


class PermissionRejectedError(Exception):
    pass
