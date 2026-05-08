# coding=utf-8
"""指标采集器测试"""

import json
import pytest

from QuantNodes.monitor.storage.models import PerformanceSnapshot
from QuantNodes.monitor.storage.repository import (
    DatabaseManager, PerformanceRepository,
)
from QuantNodes.monitor.monitor.collector import MetricsCollector


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
def collector(db):
    return MetricsCollector(PerformanceRepository(db))


class TestMetricsCollector:

    def test_collect_from_backtest(self, collector):
        stats = {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": -0.05,
            "annualized_return": 0.15,
            "annualized_volatility": 0.10,
            "win_rate": 0.55,
            "profit_factor": 1.8,
            "total_trades": 100,
        }
        snapshot = collector.collect_from_backtest("test_strategy", stats)
        assert snapshot.sharpe_ratio == 1.5
        assert snapshot.max_drawdown == -0.05

    def test_collect_with_daily_returns(self, collector):
        stats = {"sharpe_ratio": 1.0}
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        snapshot = collector.collect_from_backtest(
            "test_strategy", stats, daily_returns=returns,
        )
        assert snapshot.daily_returns is not None
        parsed = json.loads(snapshot.daily_returns)
        assert len(parsed) == 5

    def test_collect_from_lazyframe(self, collector):
        import polars as pl
        lf = pl.LazyFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "equity": [1000000, 1010000, 1005000],
        })
        snapshot = collector.collect_from_lazyframe("test_strategy", lf)
        assert snapshot is not None
        assert snapshot.sharpe_ratio is not None

    def test_collect_from_empty_lazyframe(self, collector):
        import polars as pl
        lf = pl.LazyFrame({"date": [], "equity": []})
        snapshot = collector.collect_from_lazyframe("test_strategy", lf)
        assert snapshot is not None

    def test_get_baseline_metrics(self, collector):
        from datetime import date, timedelta
        today = date.today()
        # 先保存一些快照
        for i in range(5):
            snapshot = PerformanceSnapshot(
                strategy_name="s1",
                snapshot_date=today - timedelta(days=4 - i),
                sharpe_ratio=1.0 + i * 0.1,
                daily_returns=json.dumps([0.01, -0.005, 0.02]),
            )
            collector.repo.save_snapshot(snapshot)

        baseline = collector.get_baseline_metrics("s1", baseline_days=365)
        assert baseline is not None
        assert baseline.sharpe_ratio is not None

    def test_compute_daily_returns(self):
        equity = [100, 110, 105, 115]
        returns = MetricsCollector._compute_daily_returns(equity)
        assert len(returns) == 3
        assert abs(returns[0] - 0.1) < 0.001  # (110-100)/100 = 0.1
        assert abs(returns[1] - (-5/110)) < 0.001  # (105-110)/110 ≈ -0.04545
        assert abs(returns[2] - (10/105)) < 0.001  # (115-105)/105 ≈ 0.09524

    def test_compute_statistics(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003]
        stats = MetricsCollector._compute_statistics(returns)
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats
        assert "annualized_return" in stats
