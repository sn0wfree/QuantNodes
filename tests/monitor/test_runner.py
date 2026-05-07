# -*- coding: utf-8 -*-
"""QuantNodes.monitor.scheduler.runner 单元测试"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from QuantNodes.monitor.scheduler.runner import StrategyRunner
from QuantNodes.monitor.storage.models import StrategyRun


class TestStrategyRunner:
    def test_init(self, mock_repository):
        from QuantNodes.monitor.monitor.collector import MetricsCollector
        from QuantNodes.monitor.monitor.drift import DriftDetector
        from QuantNodes.monitor.monitor.alerter import Alerter

        collector = MagicMock(spec=MetricsCollector)
        detector = MagicMock(spec=DriftDetector)
        alerter = MagicMock(spec=Alerter)

        runner = StrategyRunner(
            run_repo=mock_repository,
            metrics_collector=collector,
            drift_detector=detector,
            alerter=alerter,
        )

        assert runner.run_repo is mock_repository
        assert runner.collector is collector
        assert runner.drift_detector is detector
        assert runner.alerter is alerter

    def test_from_default_db_creates_instances(self, monitor_db_path):
        with patch("QuantNodes.monitor.scheduler.runner.DatabaseManager"):
            runner = StrategyRunner.from_default_db(monitor_db_path)
            assert runner.run_repo is not None
            assert runner.collector is not None
            assert runner.drift_detector is not None
            assert runner.alerter is not None

    def test_run_strategy_success(self, monitor_db_path):
        from QuantNodes.monitor.storage.repository import (
            DatabaseManager,
            StrategyRunRepository,
            PerformanceRepository,
            DriftAlertRepository,
        )
        from QuantNodes.monitor.monitor.collector import MetricsCollector
        from QuantNodes.monitor.monitor.drift import DriftDetector
        from QuantNodes.monitor.monitor.alerter import Alerter

        db = DatabaseManager(monitor_db_path)
        db.connect()

        run_repo = StrategyRunRepository(db)
        perf_repo = PerformanceRepository(db)
        alert_repo = DriftAlertRepository(db)
        collector = MetricsCollector(perf_repo)
        detector = DriftDetector(alert_repo, perf_repo)
        alerter = Alerter(alert_repo)

        runner = StrategyRunner(run_repo, collector, detector, alerter)

        with patch.object(runner, "_execute_backtest") as mock_backtest:
            mock_backtest.return_value = {
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "max_drawdown": -0.05,
                "annualized_return": 0.15,
            }

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("name: test\n")
                config_path = f.name

            result = runner.run_strategy(
                strategy_name="test_strategy",
                config_path=config_path,
                run_type="backtest",
            )

            assert result["status"] == "success"
            assert result["run_id"] > 0
            assert "statistics" in result

            import os
            os.unlink(config_path)

        db.close()

    def test_run_strategy_failure(self, monitor_db_path):
        from QuantNodes.monitor.storage.repository import (
            DatabaseManager,
            StrategyRunRepository,
            PerformanceRepository,
            DriftAlertRepository,
        )
        from QuantNodes.monitor.monitor.collector import MetricsCollector
        from QuantNodes.monitor.monitor.drift import DriftDetector
        from QuantNodes.monitor.monitor.alerter import Alerter

        db = DatabaseManager(monitor_db_path)
        db.connect()

        run_repo = StrategyRunRepository(db)
        perf_repo = PerformanceRepository(db)
        alert_repo = DriftAlertRepository(db)
        collector = MetricsCollector(perf_repo)
        detector = DriftDetector(alert_repo, perf_repo)
        alerter = Alerter(alert_repo)

        runner = StrategyRunner(run_repo, collector, detector, alerter)

        with patch.object(runner, "_execute_backtest") as mock_backtest:
            mock_backtest.side_effect = ValueError("Data error")

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("name: test\n")
                config_path = f.name

            result = runner.run_strategy(
                strategy_name="test_strategy",
                config_path=config_path,
                run_type="backtest",
            )

            assert result["status"] == "failed"
            assert "error" in result
            assert "Data error" in result["error"]

            import os
            os.unlink(config_path)

        db.close()

    def test_run_strategy_records_run(self, monitor_db_path):
        from QuantNodes.monitor.storage.repository import (
            DatabaseManager,
            StrategyRunRepository,
            PerformanceRepository,
            DriftAlertRepository,
        )
        from QuantNodes.monitor.monitor.collector import MetricsCollector
        from QuantNodes.monitor.monitor.drift import DriftDetector
        from QuantNodes.monitor.monitor.alerter import Alerter

        db = DatabaseManager(monitor_db_path)
        db.connect()

        run_repo = StrategyRunRepository(db)
        perf_repo = PerformanceRepository(db)
        alert_repo = DriftAlertRepository(db)
        collector = MetricsCollector(perf_repo)
        detector = DriftDetector(alert_repo, perf_repo)
        alerter = Alerter(alert_repo)

        runner = StrategyRunner(run_repo, collector, detector, alerter)

        with patch.object(runner, "_execute_backtest") as mock_backtest:
            mock_backtest.return_value = {"sharpe_ratio": 1.0}

            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("name: test\n")
                config_path = f.name

            result = runner.run_strategy(
                strategy_name="recording_test",
                config_path=config_path,
                run_type="sample_out",
                version="v1",
            )

            runs = run_repo.get_by_strategy("recording_test")
            assert len(runs) >= 1
            assert runs[0].status == "success"

            import os
            os.unlink(config_path)

        db.close()
