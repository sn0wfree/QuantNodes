# coding=utf-8
"""
Tool Context - Execution context for tools with permission checking
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio

from ..permission.service import PermissionService


@dataclass
class ToolContext:
    """Tool execution context containing permission checking and session info.

    This context is passed to tools during execution and provides
    access to permission services and abort signals.
    """
    session_id: str
    agent_id: str
    project_root: str
    permission_service: PermissionService
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)
    tool_defaults: Dict[str, Any] = field(default_factory=dict)

    async def check_permission(
        self,
        tool: str,
        permission: str,
        patterns: List[str],
        always_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if an action is permitted.

        Args:
            tool: Tool name
            permission: Permission category (e.g., "bash", "read", "edit")
            patterns: Target patterns to check
            always_patterns: Patterns that can be permanently allowed
            metadata: Additional information (e.g., diff for edits)

        Returns:
            True if allowed

        Raises:
            PermissionDeniedError: If permission is denied
            PermissionRejectedError: If user rejects
        """
        return await self.permission_service.check(
            tool=tool,
            permission=permission,
            patterns=patterns,
            always_patterns=always_patterns or [],
            metadata=metadata or {},
        )

    def is_aborted(self) -> bool:
        """Check if execution should abort"""
        return self.abort_signal.is_set()

    async def wait_for_abort(self, timeout: Optional[float] = None) -> bool:
        """Wait for abort signal.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if abort was signaled, False if timeout
        """
        try:
            await asyncio.wait_for(self.abort_signal.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class ToolContextFactory:
    """Factory for creating ToolContext instances"""

    @staticmethod
    def create(
        session_id: str,
        agent_id: str,
        project_root: str,
        permission_service: PermissionService,
        **kwargs: Any,
    ) -> ToolContext:
        """Create a new ToolContext"""
        return ToolContext(
            session_id=session_id,
            agent_id=agent_id,
            project_root=project_root,
            permission_service=permission_service,
            **kwargs,
        )