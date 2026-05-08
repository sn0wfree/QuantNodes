# -*- coding: utf-8 -*-
"""QuantNodes.monitor.monitor.dashboard 单元测试"""
from datetime import datetime, date
from unittest.mock import MagicMock

from QuantNodes.monitor.monitor.dashboard import MonitorDashboard


class TestMonitorDashboard:
    def test_init(self, mock_repository):
        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        assert dashboard.run_repo is mock_repository
        assert dashboard.perf_repo is mock_repository
        assert dashboard.alert_repo is mock_repository

    def test_get_strategy_summary_no_data(self, mock_repository):
        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_strategy_summary("nonexistent_strategy")
        assert result["strategy_name"] == "nonexistent_strategy"
        assert result["latest_run"] is None
        assert result["performance"] is None
        assert result["pending_alerts"] == 0

    def test_get_strategy_summary_with_data(self, mock_repository, sample_strategy_runs, sample_performance_snapshots):

        mock_repository.get_by_strategy.return_value = sample_strategy_runs[:1]
        perf_mock = MagicMock()
        perf_mock.sharpe_ratio = 1.5
        perf_mock.sortino_ratio = 2.0
        perf_mock.max_drawdown = -0.05
        perf_mock.annualized_return = 0.15
        perf_mock.win_rate = 0.55
        perf_mock.snapshot_date = date(2024, 1, 1)
        mock_repository.get_latest.return_value = perf_mock
        mock_repository.get_pending.return_value = []

        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_strategy_summary("strategy_a")

        assert result["strategy_name"] == "strategy_a"
        assert result["latest_run"]["status"] == "success"
        assert result["performance"]["sharpe_ratio"] == 1.5
        assert result["pending_alerts"] == 0

    def test_get_performance_history(self, mock_repository):

        perf_mock1 = MagicMock()
        perf_mock1.snapshot_date = date(2024, 1, 1)
        perf_mock1.sharpe_ratio = 1.5
        perf_mock1.max_drawdown = -0.05
        perf_mock1.annualized_return = 0.15
        perf_mock1.win_rate = 0.55

        perf_mock2 = MagicMock()
        perf_mock2.snapshot_date = date(2024, 1, 2)
        perf_mock2.sharpe_ratio = 1.6
        perf_mock2.max_drawdown = -0.04
        perf_mock2.annualized_return = 0.16
        perf_mock2.win_rate = 0.56

        mock_repository.get_history.return_value = [perf_mock1, perf_mock2]

        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_performance_history("strategy_a", days=30)

        assert len(result) == 2
        assert result[0]["sharpe_ratio"] == 1.5
        assert result[1]["sharpe_ratio"] == 1.6

    def test_get_alert_history(self, mock_repository):

        alert_mock = MagicMock()
        alert_mock.id = 1
        alert_mock.alert_type = "ks_test"
        alert_mock.severity = "warning"
        alert_mock.metric_name = "distribution"
        alert_mock.message = "Distribution drift"
        alert_mock.acknowledged = False
        alert_mock.created_at = datetime(2024, 1, 1, 10, 0, 0)

        mock_repository.get_history.return_value = [alert_mock]

        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_alert_history("strategy_a", days=30)

        assert len(result) == 1
        assert result[0]["type"] == "ks_test"
        assert result[0]["severity"] == "warning"
        assert result[0]["acknowledged"] is False

    def test_get_comparison(self, mock_repository):
        perf_mock = MagicMock()
        perf_mock.sharpe_ratio = 1.5
        perf_mock.max_drawdown = -0.05
        perf_mock.annualized_return = 0.15
        perf_mock.win_rate = 0.55

        mock_repository.get_latest.side_effect = lambda name: (
            perf_mock if name == "strategy_a" else None
        )

        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_comparison(["strategy_a", "strategy_b"])

        assert "strategy_a" in result
        assert result["strategy_a"]["sharpe_ratio"] == 1.5
        assert "strategy_b" in result
        assert result["strategy_b"] is None

    def test_get_all_strategies(self, mock_repository):
        mock_repository.db.conn.execute.return_value.fetchall.return_value = [
            {"strategy_name": "strategy_a"},
            {"strategy_name": "strategy_b"},
        ]

        dashboard = MonitorDashboard(
            run_repo=mock_repository,
            perf_repo=mock_repository,
            alert_repo=mock_repository,
        )
        result = dashboard.get_all_strategies()

        assert "strategy_a" in result
        assert "strategy_b" in result
