# coding=utf-8
"""告警生成器"""

from __future__ import annotations

from typing import List, Optional

from ..storage.models import DriftAlert
from ..storage.repository import DriftAlertRepository


class Alerter:
    """告警生成器 - 创建、查询、确认告警"""

    def __init__(self, alert_repo: DriftAlertRepository):
        self.repo = alert_repo

    def create_alert(self, alert: DriftAlert) -> int:
        """创建告警记录"""
        return self.repo.create_alert(alert)

    def get_pending_alerts(self, strategy_name: str = None) -> List[DriftAlert]:
        """获取未确认的告警"""
        return self.repo.get_pending(strategy_name)

    def acknowledge_alert(self, alert_id: int) -> None:
        """确认告警"""
        self.repo.acknowledge(alert_id)

    def get_alert_history(
        self, strategy_name: str, days: int = 30
    ) -> List[DriftAlert]:
        """获取告警历史"""
        return self.repo.get_history(strategy_name, days)

    def format_alert_message(self, alert: DriftAlert) -> str:
        """格式化告警消息"""
        severity_icon = "🔴" if alert.severity == "critical" else "🟡"
        return (
            f"{severity_icon} [{alert.severity.upper()}] {alert.strategy_name}\n"
            f"   类型: {alert.alert_type}\n"
            f"   指标: {alert.metric_name}\n"
            f"   当前值: {alert.current_value}\n"
            f"   基线值: {alert.baseline_value}\n"
            f"   消息: {alert.message}"
        )

    def get_pending_summary(self, strategy_name: str = None) -> dict:
        """获取未确认告警摘要"""
        pending = self.get_pending_alerts(strategy_name)
        return {
            "total": len(pending),
            "critical": sum(1 for a in pending if a.severity == "critical"),
            "warning": sum(1 for a in pending if a.severity == "warning"),
            "by_type": {
                "ks_test": sum(1 for a in pending if a.alert_type == "ks_test"),
                "sharpe_drop": sum(1 for a in pending if a.alert_type == "sharpe_drop"),
                "drawdown_breach": sum(1 for a in pending if a.alert_type == "drawdown_breach"),
            },
        }
