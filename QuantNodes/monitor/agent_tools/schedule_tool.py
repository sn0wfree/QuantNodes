# coding=utf-8
"""调度管理Agent工具"""

from __future__ import annotations

from typing import Any, Dict

from QuantNodes.agent.tools.base import Tool
from ..scheduler.scheduler import StrategyScheduler


class ScheduleTool(Tool):
    """策略调度管理工具"""

    def __init__(self, db_path: str = "~/.quantnodes/monitor.db"):
        self._db_path = db_path
        self._scheduler = None

    def _get_scheduler(self) -> StrategyScheduler:
        if self._scheduler is None:
            self._scheduler = StrategyScheduler(self._db_path)
            self._scheduler.start()
        return self._scheduler

    @property
    def name(self) -> str:
        return "strategy_schedule"

    @property
    def description(self) -> str:
        return "管理策略调度任务，支持添加、移除、暂停、恢复定时执行"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list", "pause", "resume"],
                    "description": "操作类型",
                },
                "strategy_name": {
                    "type": "string",
                    "description": "策略名称",
                },
                "cron": {
                    "type": "string",
                    "description": "cron表达式 (如 '0 18 * * 1-5')",
                },
                "interval_minutes": {
                    "type": "integer",
                    "description": "间隔分钟数",
                },
                "config_path": {
                    "type": "string",
                    "description": "YAML配置文件路径",
                },
            },
            "required": ["action", "strategy_name"],
        }

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, action: str, strategy_name: str,
                      cron: str = None, interval_minutes: int = None,
                      config_path: str = None, **kwargs) -> Any:
        scheduler = self._get_scheduler()

        if action == "add":
            if not config_path:
                return {"error": "config_path required for add"}
            if cron:
                job_id = scheduler.add_cron_job(strategy_name, cron, config_path)
            elif interval_minutes:
                job_id = scheduler.add_interval_job(strategy_name, interval_minutes, config_path)
            else:
                return {"error": "cron or interval_minutes required"}
            return {"status": "added", "job_id": job_id}

        elif action == "remove":
            success = scheduler.remove_job(strategy_name)
            return {"status": "removed" if success else "not_found"}

        elif action == "list":
            return {"jobs": scheduler.get_jobs()}

        elif action == "pause":
            success = scheduler.pause_job(strategy_name)
            return {"status": "paused" if success else "not_found"}

        elif action == "resume":
            success = scheduler.resume_job(strategy_name)
            return {"status": "resumed" if success else "not_found"}

        return {"error": f"Unknown action: {action}"}
