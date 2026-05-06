# coding=utf-8
"""监控数据仓储层 - SQLite CRUD操作"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List

from .models import (
    StrategyRun, PerformanceSnapshot, DriftAlert, StrategyVersion,
)

# SQLite建表DDL
_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS strategy_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    strategy_version TEXT,
    run_type TEXT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT NOT NULL,
    config_snapshot TEXT,
    statistics TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    annualized_return REAL,
    annualized_volatility REAL,
    win_rate REAL,
    profit_factor REAL,
    total_trades INTEGER,
    daily_returns TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, snapshot_date)
);

CREATE TABLE IF NOT EXISTS drift_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    metric_name TEXT,
    current_value REAL,
    baseline_value REAL,
    p_value REAL,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    version TEXT NOT NULL,
    commit_hash TEXT,
    config_snapshot TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, version)
);
"""


class DatabaseManager:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = "~/.quantnodes/monitor.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()
        return self._conn

    def _create_tables(self):
        self._conn.executescript(_CREATE_TABLES)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn


class StrategyRunRepository:
    """策略运行记录仓储"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(self, run: StrategyRun) -> int:
        cur = self.db.conn.execute(
            """INSERT INTO strategy_runs
               (strategy_name, strategy_version, run_type, start_time, end_time,
                status, config_snapshot, statistics, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run.strategy_name, run.strategy_version, run.run_type,
             run.start_time, run.end_time, run.status,
             run.config_snapshot, run.statistics, run.error_message),
        )
        self.db.conn.commit()
        return cur.lastrowid

    def get_by_id(self, run_id: int) -> Optional[StrategyRun]:
        row = self.db.conn.execute(
            "SELECT * FROM strategy_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return self._row_to_run(row) if row else None

    def get_by_strategy(self, strategy_name: str, limit: int = 50) -> List[StrategyRun]:
        rows = self.db.conn.execute(
            "SELECT * FROM strategy_runs WHERE strategy_name = ? ORDER BY created_at DESC LIMIT ?",
            (strategy_name, limit),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def update_status(self, run_id: int, status: str,
                      statistics: dict = None, error_message: str = None) -> None:
        stats_json = json.dumps(statistics) if statistics else None
        if status in ("success", "failed"):
            self.db.conn.execute(
                """UPDATE strategy_runs
                   SET status = ?, statistics = ?, error_message = ?, end_time = ?
                   WHERE id = ?""",
                (status, stats_json, error_message, datetime.now(), run_id),
            )
        else:
            self.db.conn.execute(
                "UPDATE strategy_runs SET status = ? WHERE id = ?",
                (status, run_id),
            )
        self.db.conn.commit()

    def delete_old(self, strategy_name: str, keep_count: int = 100) -> int:
        # SQLite 不支持 OFFSET in DELETE subquery, 使用 LIMIT 方式
        cur = self.db.conn.execute(
            """DELETE FROM strategy_runs WHERE id NOT IN (
                   SELECT id FROM strategy_runs
                   WHERE strategy_name = ?
                   ORDER BY created_at DESC
                   LIMIT ?
               ) AND strategy_name = ?""",
            (strategy_name, keep_count, strategy_name),
        )
        self.db.conn.commit()
        return cur.rowcount

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> StrategyRun:
        return StrategyRun(
            id=row["id"],
            strategy_name=row["strategy_name"],
            strategy_version=row["strategy_version"],
            run_type=row["run_type"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            config_snapshot=row["config_snapshot"],
            statistics=row["statistics"],
            error_message=row["error_message"],
            created_at=row["created_at"],
        )


class PerformanceRepository:
    """绩效快照仓储"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_snapshot(self, snapshot: PerformanceSnapshot) -> int:
        cur = self.db.conn.execute(
            """INSERT OR REPLACE INTO performance_snapshots
               (strategy_name, snapshot_date, sharpe_ratio, sortino_ratio,
                max_drawdown, annualized_return, annualized_volatility,
                win_rate, profit_factor, total_trades, daily_returns)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot.strategy_name, snapshot.snapshot_date,
             snapshot.sharpe_ratio, snapshot.sortino_ratio,
             snapshot.max_drawdown, snapshot.annualized_return,
             snapshot.annualized_volatility, snapshot.win_rate,
             snapshot.profit_factor, snapshot.total_trades,
             snapshot.daily_returns),
        )
        self.db.conn.commit()
        return cur.lastrowid

    def get_latest(self, strategy_name: str) -> Optional[PerformanceSnapshot]:
        row = self.db.conn.execute(
            """SELECT * FROM performance_snapshots
               WHERE strategy_name = ? ORDER BY snapshot_date DESC LIMIT 1""",
            (strategy_name,),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def get_history(self, strategy_name: str, days: int = 30) -> List[PerformanceSnapshot]:
        cutoff = date.today() - timedelta(days=days)
        rows = self.db.conn.execute(
            """SELECT * FROM performance_snapshots
               WHERE strategy_name = ? AND snapshot_date >= ?
               ORDER BY snapshot_date""",
            (strategy_name, cutoff.isoformat()),
        ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def get_baseline(self, strategy_name: str, days: int = 252) -> Optional[PerformanceSnapshot]:
        cutoff = date.today() - timedelta(days=days)
        rows = self.db.conn.execute(
            """SELECT * FROM performance_snapshots
               WHERE strategy_name = ? AND snapshot_date >= ?
               ORDER BY snapshot_date""",
            (strategy_name, cutoff.isoformat()),
        ).fetchall()
        if not rows:
            return None
        snapshots = [self._row_to_snapshot(r) for r in rows]
        return self._average_snapshots(snapshots)

    @staticmethod
    def _average_snapshots(snapshots: List[PerformanceSnapshot]) -> PerformanceSnapshot:
        n = len(snapshots)
        s = snapshots[0]
        return PerformanceSnapshot(
            strategy_name=s.strategy_name,
            snapshot_date=s.snapshot_date,
            sharpe_ratio=sum(s.sharpe_ratio or 0 for s in snapshots) / n,
            sortino_ratio=sum(s.sortino_ratio or 0 for s in snapshots) / n,
            max_drawdown=sum(s.max_drawdown or 0 for s in snapshots) / n,
            annualized_return=sum(s.annualized_return or 0 for s in snapshots) / n,
            annualized_volatility=sum(s.annualized_volatility or 0 for s in snapshots) / n,
            win_rate=sum(s.win_rate or 0 for s in snapshots) / n,
            profit_factor=sum(s.profit_factor or 0 for s in snapshots) / n,
            total_trades=sum(s.total_trades or 0 for s in snapshots) // n,
        )

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            id=row["id"],
            strategy_name=row["strategy_name"],
            snapshot_date=row["snapshot_date"],
            sharpe_ratio=row["sharpe_ratio"],
            sortino_ratio=row["sortino_ratio"],
            max_drawdown=row["max_drawdown"],
            annualized_return=row["annualized_return"],
            annualized_volatility=row["annualized_volatility"],
            win_rate=row["win_rate"],
            profit_factor=row["profit_factor"],
            total_trades=row["total_trades"],
            daily_returns=row["daily_returns"],
            created_at=row["created_at"],
        )


class DriftAlertRepository:
    """漂移告警仓储"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_alert(self, alert: DriftAlert) -> int:
        cur = self.db.conn.execute(
            """INSERT INTO drift_alerts
               (strategy_name, alert_type, severity, metric_name,
                current_value, baseline_value, p_value, message, acknowledged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert.strategy_name, alert.alert_type, alert.severity,
             alert.metric_name, alert.current_value, alert.baseline_value,
             alert.p_value, alert.message, alert.acknowledged),
        )
        self.db.conn.commit()
        return cur.lastrowid

    def get_pending(self, strategy_name: str = None) -> List[DriftAlert]:
        if strategy_name:
            rows = self.db.conn.execute(
                """SELECT * FROM drift_alerts
                   WHERE strategy_name = ? AND acknowledged = 0
                   ORDER BY created_at DESC""",
                (strategy_name,),
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                """SELECT * FROM drift_alerts
                   WHERE acknowledged = 0 ORDER BY created_at DESC"""
            ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def acknowledge(self, alert_id: int) -> None:
        self.db.conn.execute(
            "UPDATE drift_alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        self.db.conn.commit()

    def get_history(self, strategy_name: str, days: int = 30) -> List[DriftAlert]:
        cutoff = datetime.now() - timedelta(days=days)
        rows = self.db.conn.execute(
            """SELECT * FROM drift_alerts
               WHERE strategy_name = ? AND created_at >= ?
               ORDER BY created_at DESC""",
            (strategy_name, cutoff),
        ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> DriftAlert:
        return DriftAlert(
            id=row["id"],
            strategy_name=row["strategy_name"],
            alert_type=row["alert_type"],
            severity=row["severity"],
            metric_name=row["metric_name"],
            current_value=row["current_value"],
            baseline_value=row["baseline_value"],
            p_value=row["p_value"],
            message=row["message"],
            acknowledged=bool(row["acknowledged"]),
            created_at=row["created_at"],
        )


class VersionRepository:
    """策略版本仓储"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_version(self, version: StrategyVersion) -> int:
        cur = self.db.conn.execute(
            """INSERT OR REPLACE INTO strategy_versions
               (strategy_name, version, commit_hash, config_snapshot, description)
               VALUES (?, ?, ?, ?, ?)""",
            (version.strategy_name, version.version,
             version.commit_hash, version.config_snapshot,
             version.description),
        )
        self.db.conn.commit()
        return cur.lastrowid

    def get_version(self, strategy_name: str, version: str) -> Optional[StrategyVersion]:
        row = self.db.conn.execute(
            """SELECT * FROM strategy_versions
               WHERE strategy_name = ? AND version = ?""",
            (strategy_name, version),
        ).fetchone()
        return self._row_to_version(row) if row else None

    def list_versions(self, strategy_name: str) -> List[StrategyVersion]:
        rows = self.db.conn.execute(
            """SELECT * FROM strategy_versions
               WHERE strategy_name = ? ORDER BY id DESC""",
            (strategy_name,),
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def get_latest(self, strategy_name: str) -> Optional[StrategyVersion]:
        row = self.db.conn.execute(
            """SELECT * FROM strategy_versions
               WHERE strategy_name = ? ORDER BY id DESC LIMIT 1""",
            (strategy_name,),
        ).fetchone()
        return self._row_to_version(row) if row else None

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> StrategyVersion:
        return StrategyVersion(
            id=row["id"],
            strategy_name=row["strategy_name"],
            version=row["version"],
            commit_hash=row["commit_hash"],
            config_snapshot=row["config_snapshot"],
            description=row["description"],
            created_at=row["created_at"],
        )
