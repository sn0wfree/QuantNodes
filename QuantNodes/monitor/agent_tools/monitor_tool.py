# coding=utf-8
"""监控查询Agent工具"""

from __future__ import annotations

from typing import Any, Dict

from QuantNodes.agent.tools.base import Tool
from ..storage.repository import DatabaseManager
from ..monitor.dashboard import MonitorDashboard


class MonitorTool(Tool):
    """策略监控查询工具"""

    def __init__(self, db_path: str = "~/.quantnodes/monitor.db"):
        self._db_path = db_path
        self._dashboard = None

    def _get_dashboard(self) -> MonitorDashboard:
        if self._dashboard is None:
            db = DatabaseManager(self._db_path)
            db.connect()
            from ..storage.repository import (
                StrategyRunRepository, PerformanceRepository, DriftAlertRepository,
            )
            self._dashboard = MonitorDashboard(
                StrategyRunRepository(db),
                PerformanceRepository(db),
                DriftAlertRepository(db),
            )
        return self._dashboard

    @property
    def name(self) -> str:
        return "strategy_monitor"

    @property
    def description(self) -> str:
        return "查询策略监控状态，包括绩效指标、漂移告警、历史记录等"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "history", "alerts", "compare"],
                    "description": "操作类型",
                },
                "strategy_name": {
                    "type": "string",
                    "description": "策略名称",
                },
                "days": {
                    "type": "integer",
                    "description": "查询天数 (默认30)",
                    "default": 30,
                },
                "strategy_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "对比的策略名称列表 (仅compare操作)",
                },
            },
            "required": ["action"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, action: str, strategy_name: str = None,
                      days: int = 30, strategy_names: list = None, **kwargs) -> Any:
        dashboard = self._get_dashboard()

        if action == "status":
            if not strategy_name:
                return {"error": "strategy_name required for status"}
            return dashboard.get_strategy_summary(strategy_name)

        elif action == "history":
            if not strategy_name:
                return {"error": "strategy_name required for history"}
            return dashboard.get_performance_history(strategy_name, days)

        elif action == "alerts":
            if not strategy_name:
                return {"error": "strategy_name required for alerts"}
            return dashboard.get_alert_history(strategy_name, days)

        elif action == "compare":
            if not strategy_names:
                return {"error": "strategy_names required for compare"}
            return dashboard.get_comparison(strategy_names)

        return {"error": f"Unknown action: {action}"}
