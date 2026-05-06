# coding=utf-8
"""监控数据查询接口"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from ..storage.repository import (
    StrategyRunRepository,
    PerformanceRepository,
    DriftAlertRepository,
)


class MonitorDashboard:
    """监控数据查询接口"""

    def __init__(
        self,
        run_repo: StrategyRunRepository,
        perf_repo: PerformanceRepository,
        alert_repo: DriftAlertRepository,
    ):
        self.run_repo = run_repo
        self.perf_repo = perf_repo
        self.alert_repo = alert_repo

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略概览"""
        latest_run = None
        runs = self.run_repo.get_by_strategy(strategy_name, limit=1)
        if runs:
            latest_run = {
                "status": runs[0].status,
                "start_time": str(runs[0].start_time) if runs[0].start_time else None,
                "end_time": str(runs[0].end_time) if runs[0].end_time else None,
            }

        perf = self.perf_repo.get_latest(strategy_name)
        perf_dict = None
        if perf:
            perf_dict = {
                "sharpe_ratio": perf.sharpe_ratio,
                "sortino_ratio": perf.sortino_ratio,
                "max_drawdown": perf.max_drawdown,
                "annualized_return": perf.annualized_return,
                "win_rate": perf.win_rate,
                "snapshot_date": str(perf.snapshot_date) if perf.snapshot_date else None,
            }

        pending_alerts = self.alert_repo.get_pending(strategy_name)

        return {
            "strategy_name": strategy_name,
            "latest_run": latest_run,
            "performance": perf_dict,
            "pending_alerts": len(pending_alerts),
        }

    def get_performance_history(
        self, strategy_name: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取绩效历史"""
        snapshots = self.perf_repo.get_history(strategy_name, days)
        return [
            {
                "date": str(s.snapshot_date),
                "sharpe_ratio": s.sharpe_ratio,
                "max_drawdown": s.max_drawdown,
                "annualized_return": s.annualized_return,
                "win_rate": s.win_rate,
            }
            for s in snapshots
        ]

    def get_alert_history(
        self, strategy_name: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取告警历史"""
        alerts = self.alert_repo.get_history(strategy_name, days)
        return [
            {
                "id": a.id,
                "type": a.alert_type,
                "severity": a.severity,
                "metric": a.metric_name,
                "message": a.message,
                "acknowledged": a.acknowledged,
                "created_at": str(a.created_at) if a.created_at else None,
            }
            for a in alerts
        ]

    def get_comparison(self, strategy_names: List[str]) -> Dict[str, Any]:
        """多策略对比"""
        result = {}
        for name in strategy_names:
            perf = self.perf_repo.get_latest(name)
            if perf:
                result[name] = {
                    "sharpe_ratio": perf.sharpe_ratio,
                    "max_drawdown": perf.max_drawdown,
                    "annualized_return": perf.annualized_return,
                    "win_rate": perf.win_rate,
                }
            else:
                result[name] = None
        return result

    def get_all_strategies(self) -> List[str]:
        """获取所有策略名称"""
        rows = self.run_repo.db.conn.execute(
            "SELECT DISTINCT strategy_name FROM strategy_runs"
        ).fetchall()
        return [r["strategy_name"] for r in rows]
