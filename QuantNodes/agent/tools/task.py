# coding=utf-8
"""
Task 工具

提供简单的任务管理功能，支持创建、更新、列表任务。
任务数据存储在 workspace/.quant_agent/tasks.json。
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool


class TaskTool(Tool):
    """任务管理工具

    创建、更新、列表任务。数据持久化到 JSON 文件。
    """

    MAX_TASKS = 100

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self._tasks_file = self.workspace / ".quant_agent" / "tasks.json"

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return (
            "任务管理工具：创建、更新、列表任务。"
            "支持按状态筛选，任务持久化到 JSON 文件。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_task", "update_task", "list_tasks"],
                    "description": "操作类型",
                },
                "title": {
                    "type": "string",
                    "description": "任务标题（create_task 时必需）",
                },
                "description": {
                    "type": "string",
                    "description": "任务描述",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "任务优先级（默认 medium）",
                    "default": "medium",
                },
                "task_id": {
                    "type": "string",
                    "description": "任务 ID（update_task 时必需）",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "cancelled"],
                    "description": "任务状态（update_task 时使用）",
                },
            },
            "required": ["action"],
        }

    @property
    def read_only(self) -> bool:
        return False

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """从 JSON 文件加载任务"""
        if not self._tasks_file.exists():
            return []
        try:
            return json.loads(self._tasks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return []

    def _save_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """保存任务到 JSON 文件"""
        self._tasks_file.parent.mkdir(parents=True, exist_ok=True)
        self._tasks_file.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def execute(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "create_task": self._create_task,
            "update_task": self._update_task,
            "list_tasks": self._list_tasks,
        }
        fn = dispatch.get(action)
        if not fn:
            return {"error": f"Unknown action: {action}"}
        return await fn(**kwargs)

    async def _create_task(
        self,
        title: str = "",
        description: str = "",
        priority: str = "medium",
        **kw,
    ) -> Dict[str, Any]:
        if not title:
            return {"error": "title is required"}

        tasks = self._load_tasks()
        if len(tasks) >= self.MAX_TASKS:
            return {"error": f"Task limit reached ({self.MAX_TASKS})"}

        now = datetime.now(timezone.utc).isoformat()
        task = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "description": description,
            "status": "pending",
            "priority": priority,
            "created_at": now,
            "updated_at": now,
        }
        tasks.append(task)
        self._save_tasks(tasks)

        return {"status": "ok", "task": task}

    async def _update_task(
        self,
        task_id: str = "",
        status: Optional[str] = None,
        description: Optional[str] = None,
        **kw,
    ) -> Dict[str, Any]:
        if not task_id:
            return {"error": "task_id is required"}

        tasks = self._load_tasks()
        for task in tasks:
            if task["id"] == task_id:
                if status:
                    task["status"] = status
                if description is not None:
                    task["description"] = description
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_tasks(tasks)
                return {"status": "ok", "task": task}

        return {"error": f"Task not found: {task_id}"}

    async def _list_tasks(self, status: Optional[str] = None, **kw) -> Dict[str, Any]:
        tasks = self._load_tasks()
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return {"tasks": tasks, "total": len(tasks)}
