# -*- coding: utf-8 -*-
"""QuantNodes.monitor.monitor.alerter 单元测试"""

from QuantNodes.monitor.monitor.alerter import Alerter


class TestAlerter:
    def test_init(self, monitor_db_path):
        from QuantNodes.monitor.storage.repository import DatabaseManager, DriftAlertRepository

        db = DatabaseManager(monitor_db_path)
        db.connect()
        repo = DriftAlertRepository(db)
        alerter = Alerter(repo)
        assert alerter.repo is repo
        db.close()

    def test_create_alert(self, monitor_db):
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="test_strategy",
            alert_type="ks_test",
            severity="warning",
            metric_name="distribution",
            current_value=0.15,
            baseline_value=0.05,
            message="Distribution drift detected",
        )
        alert_id = alerter.create_alert(alert)
        assert alert_id > 0

    def test_get_pending_alerts(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="test_strategy",
            alert_type="sharpe_drop",
            severity="critical",
            metric_name="sharpe_ratio",
            current_value=0.5,
            baseline_value=1.5,
            message="Sharpe ratio dropped significantly",
            acknowledged=False,
        )
        alerter.create_alert(alert)

        pending = alerter.get_pending_alerts("test_strategy")
        assert len(pending) >= 1
        assert all(not a.acknowledged for a in pending)

    def test_get_pending_alerts_by_strategy(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        for name in ["strategy_a", "strategy_b"]:
            alert = DriftAlert(
                strategy_name=name,
                alert_type="ks_test",
                severity="warning",
                metric_name="distribution",
                current_value=0.1,
                baseline_value=0.05,
                message="Drift",
                acknowledged=False,
            )
            alerter.create_alert(alert)

        pending_a = alerter.get_pending_alerts("strategy_a")
        pending_b = alerter.get_pending_alerts("strategy_b")
        assert all(a.strategy_name == "strategy_a" for a in pending_a)
        assert all(a.strategy_name == "strategy_b" for a in pending_b)

    def test_acknowledge_alert(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="test_strategy",
            alert_type="drawdown_breach",
            severity="critical",
            metric_name="max_drawdown",
            current_value=-0.15,
            baseline_value=-0.05,
            message="Max drawdown exceeded threshold",
            acknowledged=False,
        )
        alert_id = alerter.create_alert(alert)

        alerter.acknowledge_alert(alert_id)
        pending = alerter.get_pending_alerts("test_strategy")
        assert all(a.id != alert_id or a.acknowledged for a in pending)

    def test_get_alert_history(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="test_strategy",
            alert_type="ks_test",
            severity="warning",
            metric_name="distribution",
            current_value=0.1,
            baseline_value=0.05,
            message="Historical alert",
        )
        alerter.create_alert(alert)

        history = alerter.get_alert_history("test_strategy", days=30)
        assert len(history) >= 1

    def test_format_alert_message_critical(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="test_strategy",
            alert_type="ks_test",
            severity="critical",
            metric_name="distribution",
            current_value=0.2,
            baseline_value=0.05,
            message="Severe drift detected",
        )
        msg = alerter.format_alert_message(alert)
        assert "CRITICAL" in msg
        assert "test_strategy" in msg
        assert "0.2" in msg

    def test_format_alert_message_warning(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="strategy_x",
            alert_type="sharpe_drop",
            severity="warning",
            metric_name="sharpe_ratio",
            current_value=0.8,
            baseline_value=1.5,
            message="Sharpe dropped",
        )
        msg = alerter.format_alert_message(alert)
        assert "WARNING" in msg
        assert "strategy_x" in msg

    def test_get_pending_summary(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        for i, (severity, alert_type) in enumerate([
            ("critical", "ks_test"),
            ("warning", "sharpe_drop"),
            ("warning", "ks_test"),
        ]):
            alert = DriftAlert(
                strategy_name="test_strategy",
                alert_type=alert_type,
                severity=severity,
                metric_name="metric",
                current_value=0.1,
                baseline_value=0.05,
                message=f"Alert {i}",
            )
            alerter.create_alert(alert)

        summary = alerter.get_pending_summary("test_strategy")
        assert summary["total"] >= 3
        assert summary["critical"] >= 1
        assert summary["warning"] >= 2
        assert "ks_test" in summary["by_type"]
        assert "sharpe_drop" in summary["by_type"]

    def test_get_pending_summary_all_strategies(self, monitor_db):
        from QuantNodes.monitor.storage.repository import DriftAlertRepository
        from QuantNodes.monitor.storage.models import DriftAlert

        repo = monitor_db
        alert_repo = DriftAlertRepository(repo)
        alerter = Alerter(alert_repo)

        alert = DriftAlert(
            strategy_name="strategy_y",
            alert_type="ks_test",
            severity="warning",
            metric_name="distribution",
            current_value=0.1,
            baseline_value=0.05,
            message="Alert",
        )
        alerter.create_alert(alert)

        summary = alerter.get_pending_summary()
        assert summary["total"] >= 1
