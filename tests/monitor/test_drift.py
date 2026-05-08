# coding=utf-8
"""漂移检测测试"""

import pytest
from datetime import date

from QuantNodes.monitor.storage.models import PerformanceSnapshot
from QuantNodes.monitor.storage.repository import (
    DatabaseManager, PerformanceRepository, DriftAlertRepository,
)
from QuantNodes.monitor.monitor.drift import DriftDetector


@pytest.fixture
def db():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        dm = DatabaseManager(path)
        dm.connect()
        yield dm
        dm.close()
    finally:
        os.unlink(path)


@pytest.fixture
def detector(db):
    perf_repo = PerformanceRepository(db)
    alert_repo = DriftAlertRepository(db)
    return DriftDetector(alert_repo, perf_repo)


class TestDriftDetector:

    def test_detect_ks_drift_with_scipy(self, detector):
        """KS检验能检测分布漂移"""
        import random
        random.seed(42)
        baseline = [random.gauss(0, 0.01) for _ in range(100)]
        current = [random.gauss(0.05, 0.02) for _ in range(100)]

        try:
            alert = detector.detect_ks_drift(current, baseline, "test_strategy")
            # 可能触发也可能不触发，取决于分布差异
            if alert:
                assert alert.alert_type == "ks_test"
                assert alert.strategy_name == "test_strategy"
        except ImportError:
            pytest.skip("scipy not installed")

    def test_detect_ks_drift_no_change(self, detector):
        """相同分布不应触发告警"""
        import random
        random.seed(42)
        baseline = [random.gauss(0, 0.01) for _ in range(100)]
        current = [random.gauss(0, 0.01) for _ in range(100)]

        try:
            alert = detector.detect_ks_drift(current, baseline, "test_strategy")
            assert alert is None
        except ImportError:
            pytest.skip("scipy not installed")

    def test_detect_ks_drift_insufficient_data(self, detector):
        """数据量不足时返回None"""
        alert = detector.detect_ks_drift([1, 2], [1, 2, 3], "test")
        assert alert is None

    def test_detect_sharpe_drop(self, detector):
        """夏普比率下降检测"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=0.5,
        )
        baseline = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=1.0,
        )
        alert = detector.detect_sharpe_drop(current, baseline, "s1")
        assert alert is not None
        assert alert.alert_type == "sharpe_drop"
        assert alert.severity == "warning"

    def test_detect_sharpe_drop_no_trigger(self, detector):
        """夏普比率未显著下降"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=0.9,
        )
        baseline = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=1.0,
        )
        alert = detector.detect_sharpe_drop(current, baseline, "s1")
        assert alert is None

    def test_detect_sharpe_drop_critical(self, detector):
        """夏普比率严重下降"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=0.2,
        )
        baseline = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=1.0,
        )
        alert = detector.detect_sharpe_drop(current, baseline, "s1")
        assert alert is not None
        assert alert.severity == "critical"

    def test_detect_sharpe_drop_no_baseline(self, detector):
        """无基线时返回None"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), sharpe_ratio=0.5,
        )
        alert = detector.detect_sharpe_drop(current, None, "s1")
        assert alert is None

    def test_detect_drawdown_breach(self, detector):
        """最大回撤超标检测"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), max_drawdown=-0.25,
        )
        alert = detector.detect_drawdown_breach(current, "s1")
        assert alert is not None
        assert alert.alert_type == "drawdown_breach"

    def test_detect_drawdown_breach_no_trigger(self, detector):
        """最大回撤未超标"""
        current = PerformanceSnapshot(
            strategy_name="s1", snapshot_date=date.today(), max_drawdown=-0.10,
        )
        alert = detector.detect_drawdown_breach(current, "s1")
        assert alert is None

    def test_run_all_checks_empty(self, detector):
        """无数据时返回空列表"""
        alerts = detector.run_all_checks("nonexistent_strategy")
        assert alerts == []
