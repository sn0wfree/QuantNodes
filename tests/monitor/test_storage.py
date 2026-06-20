# coding=utf-8
"""存储层测试"""

import pytest
import tempfile
import os
from datetime import date

from QuantNodes.monitor.storage.models import (
    StrategyRun, PerformanceSnapshot, DriftAlert, StrategyVersion,
)
from QuantNodes.monitor.storage.repository import (
    DatabaseManager, StrategyRunRepository, PerformanceRepository,
    DriftAlertRepository, VersionRepository,
)


@pytest.fixture
def db():
    """临时数据库 fixture"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        dm = DatabaseManager(db_path)
        dm.connect()
        yield dm
        dm.close()
    finally:
        os.unlink(db_path)


@pytest.fixture
def run_repo(db):
    return StrategyRunRepository(db)


@pytest.fixture
def perf_repo(db):
    return PerformanceRepository(db)


@pytest.fixture
def alert_repo(db):
    return DriftAlertRepository(db)


@pytest.fixture
def version_repo(db):
    return VersionRepository(db)


class TestDatabaseManager:

    def test_connect_creates_tables(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "strategy_runs" in table_names
        assert "performance_snapshots" in table_names
        assert "drift_alerts" in table_names
        assert "strategy_versions" in table_names

    def test_context_manager(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            with DatabaseManager(path) as db:
                assert db.conn is not None
        finally:
            os.unlink(path)


class TestStrategyRunRepository:

    def test_create_and_get(self, run_repo):
        run = StrategyRun(
            strategy_name="test_strategy",
            run_type="backtest",
            status="success",
        )
        run_id = run_repo.create(run)
        assert run_id > 0

        fetched = run_repo.get_by_id(run_id)
        assert fetched is not None
        assert fetched.strategy_name == "test_strategy"
        assert fetched.status == "success"

    def test_get_by_strategy(self, run_repo):
        for i in range(3):
            run_repo.create(StrategyRun(
                strategy_name="s1", run_type="backtest", status="success",
            ))
        runs = run_repo.get_by_strategy("s1")
        assert len(runs) == 3

    def test_update_status(self, run_repo):
        run_id = run_repo.create(StrategyRun(
            strategy_name="s1", run_type="backtest", status="running",
        ))
        run_repo.update_status(run_id, "success", statistics={"sharpe": 1.5})
        fetched = run_repo.get_by_id(run_id)
        assert fetched.status == "success"

    def test_delete_old(self, run_repo):
        for i in range(5):
            run_repo.create(StrategyRun(
                strategy_name="s1", run_type="backtest", status="success",
            ))
        run_repo.delete_old("s1", keep_count=2)
        remaining = run_repo.get_by_strategy("s1", limit=100)
        assert len(remaining) == 2


class TestPerformanceRepository:

    def test_save_and_get_latest(self, perf_repo):
        snap = PerformanceSnapshot(
            strategy_name="s1",
            snapshot_date=date.today(),
            sharpe_ratio=1.5,
            max_drawdown=-0.05,
        )
        perf_repo.save_snapshot(snap)

        latest = perf_repo.get_latest("s1")
        assert latest is not None
        assert latest.sharpe_ratio == 1.5
        assert latest.max_drawdown == -0.05

    def test_get_history(self, perf_repo):
        from datetime import date, timedelta
        today = date.today()
        for i in range(5):
            snap = PerformanceSnapshot(
                strategy_name="s1",
                snapshot_date=today - timedelta(days=4 - i),
                sharpe_ratio=float(i),
            )
            perf_repo.save_snapshot(snap)
        history = perf_repo.get_history("s1", days=365)
        assert len(history) == 5

    def test_get_baseline(self, perf_repo):
        from datetime import date, timedelta
        today = date.today()
        for i in range(10):
            snap = PerformanceSnapshot(
                strategy_name="s1",
                snapshot_date=today - timedelta(days=9 - i),
                sharpe_ratio=1.0 + i * 0.1,
            )
            perf_repo.save_snapshot(snap)
        baseline = perf_repo.get_baseline("s1", days=365)
        assert baseline is not None
        assert baseline.sharpe_ratio is not None


class TestDriftAlertRepository:

    def test_create_and_get_pending(self, alert_repo):
        alert = DriftAlert(
            strategy_name="s1",
            alert_type="ks_test",
            severity="warning",
            message="test alert",
        )
        alert_id = alert_repo.create_alert(alert)
        assert alert_id > 0

        pending = alert_repo.get_pending("s1")
        assert len(pending) == 1
        assert pending[0].alert_type == "ks_test"

    def test_acknowledge(self, alert_repo):
        alert_repo.create_alert(DriftAlert(
            strategy_name="s1", alert_type="sharpe_drop", severity="critical",
        ))
        pending = alert_repo.get_pending("s1")
        assert len(pending) == 1

        alert_repo.acknowledge(pending[0].id)
        pending_after = alert_repo.get_pending("s1")
        assert len(pending_after) == 0

    def test_get_history(self, alert_repo):
        alert_repo.create_alert(DriftAlert(
            strategy_name="s1", alert_type="drawdown_breach", severity="warning",
        ))
        history = alert_repo.get_history("s1", days=30)
        assert len(history) == 1


class TestVersionRepository:

    def test_save_and_get(self, version_repo):
        sv = StrategyVersion(
            strategy_name="s1",
            version="v1",
            config_snapshot="name: test",
            description="initial version",
        )
        version_repo.save_version(sv)

        fetched = version_repo.get_version("s1", "v1")
        assert fetched is not None
        assert fetched.config_snapshot == "name: test"

    def test_list_versions(self, version_repo):
        for i in range(3):
            version_repo.save_version(StrategyVersion(
                strategy_name="s1",
                version=f"v{i+1}",
                config_snapshot=f"name: test_{i}",
            ))
        versions = version_repo.list_versions("s1")
        assert len(versions) == 3

    def test_get_latest(self, version_repo):
        version_repo.save_version(StrategyVersion(
            strategy_name="s1", version="v1", config_snapshot="a",
        ))
        version_repo.save_version(StrategyVersion(
            strategy_name="s1", version="v2", config_snapshot="b",
        ))
        latest = version_repo.get_latest("s1")
        assert latest.version == "v2"
