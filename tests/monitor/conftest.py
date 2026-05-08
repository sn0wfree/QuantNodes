# -*- coding: utf-8 -*-
"""Monitor 子系统测试 fixtures"""
import pytest


@pytest.fixture
def monitor_db_path(tmp_path):
    """Monitor 临时数据库路径"""
    return str(tmp_path / "monitor_test.db")


@pytest.fixture
def monitor_db(monitor_db_path):
    """Monitor 临时数据库（已连接并初始化 schema）

    返回 DatabaseManager 实例
    """
    from QuantNodes.monitor.storage.repository import DatabaseManager

    db = DatabaseManager(monitor_db_path)
    db.connect()
    yield db
    db.close()


@pytest.fixture
def mock_dashboard():
    """Mock MonitorDashboard，用于工具测试"""
    from unittest.mock import MagicMock

    dashboard = MagicMock()
    dashboard.get_strategy_summary.return_value = {
        "strategy_name": "test_strategy",
        "latest_run": {"status": "success", "start_time": "2024-01-01 10:00:00"},
        "performance": {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "annualized_return": 0.15,
        },
        "pending_alerts": 0,
    }
    dashboard.get_performance_history.return_value = [
        {"date": "2024-01-01", "sharpe_ratio": 1.5, "max_drawdown": -0.05}
    ]
    dashboard.get_alert_history.return_value = []
    dashboard.get_comparison.return_value = {
        "test_strategy": {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "annualized_return": 0.15,
        }
    }
    return dashboard


@pytest.fixture
def mock_repository():
    """Mock 仓库，用于隔离测试"""
    from unittest.mock import MagicMock

    repo = MagicMock()
    repo.get_by_strategy.return_value = []
    repo.get_latest.return_value = None
    repo.get_pending.return_value = []
    repo.get_history.return_value = []
    return repo


@pytest.fixture
def sample_strategy_runs():
    """样本策略运行记录"""
    from datetime import datetime
    from QuantNodes.monitor.storage.models import StrategyRun

    return [
        StrategyRun(
            id=1,
            strategy_name="strategy_a",
            run_type="backtest",
            status="success",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 1, 0),
            strategy_version="v1",
            config_snapshot="name: strategy_a\n",
        ),
        StrategyRun(
            id=2,
            strategy_name="strategy_a",
            run_type="backtest",
            status="failed",
            start_time=datetime(2024, 1, 2, 10, 0, 0),
            end_time=datetime(2024, 1, 2, 10, 0, 30),
            strategy_version="v1",
            error_message="Data error",
        ),
    ]


@pytest.fixture
def sample_performance_snapshots():
    """样本绩效快照"""
    from datetime import date
    from QuantNodes.monitor.storage.models import PerformanceSnapshot

    return [
        PerformanceSnapshot(
            id=1,
            strategy_name="strategy_a",
            snapshot_date=date(2024, 1, 1),
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.05,
            annualized_return=0.15,
            win_rate=0.55,
        ),
        PerformanceSnapshot(
            id=2,
            strategy_name="strategy_a",
            snapshot_date=date(2024, 1, 2),
            sharpe_ratio=1.6,
            sortino_ratio=2.1,
            max_drawdown=-0.04,
            annualized_return=0.16,
            win_rate=0.56,
        ),
    ]
