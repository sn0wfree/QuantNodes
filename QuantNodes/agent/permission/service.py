# coding=utf-8
"""
权限服务
"""

import asyncio
from typing import Dict, Callable, Awaitable
from dataclasses import dataclass

from .models import (
    Action, PermissionRule, PermissionRequest, PermissionReply, Ruleset,
    PermissionDeniedError, PermissionRejectedError
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
        self._approved: Ruleset = []
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

            if self._reply_handler is None:
                raise PermissionDeniedError(
                    f"No reply handler configured, permission denied: {permission} on {pattern}"
                )

            request = PermissionRequest(
                id=f"{tool}_{pattern}",
                session_id="",
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
                for p in always_patterns:
                    self._approved.append(PermissionRule(
                        permission=permission,
                        pattern=p,
                        action=Action.ALLOW,
                    ))

        return True

    def get_disabled_tools(self) -> set[str]:
        """获取被禁用的工具列表"""
        disabled = set()
        for rule in self._ruleset:
            if rule.pattern == "*" and rule.action == Action.DENY:
                disabled.add(rule.permission)
        return disabled
