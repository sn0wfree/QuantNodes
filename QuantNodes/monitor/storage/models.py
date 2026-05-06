# coding=utf-8
"""监控数据模型 - SQLite表结构定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List


@dataclass
class StrategyRun:
    """策略运行记录"""
    strategy_name: str
    run_type: str  # 'backtest' | 'live' | 'sample_out'
    status: str  # 'running' | 'success' | 'failed'
    strategy_version: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    config_snapshot: Optional[str] = None
    statistics: Optional[str] = None  # JSON字符串
    error_message: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """绩效快照"""
    strategy_name: str
    snapshot_date: date
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    daily_returns: Optional[str] = None  # JSON数组
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class DriftAlert:
    """漂移告警"""
    strategy_name: str
    alert_type: str  # 'ks_test' | 'sharpe_drop' | 'drawdown_breach'
    severity: str  # 'warning' | 'critical'
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    baseline_value: Optional[float] = None
    p_value: Optional[float] = None
    message: Optional[str] = None
    acknowledged: bool = False
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class StrategyVersion:
    """策略版本"""
    strategy_name: str
    version: str
    config_snapshot: str  # YAML配置内容
    commit_hash: Optional[str] = None
    description: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
