# coding=utf-8
"""绩效指标采集器"""

from __future__ import annotations

import json
from datetime import date
from typing import Optional, Dict, Any, List

import polars as pl

from ..storage.models import PerformanceSnapshot
from ..storage.repository import PerformanceRepository


class MetricsCollector:
    """绩效指标采集器 - 从回测结果或实盘数据采集绩效指标"""

    def __init__(self, performance_repo: PerformanceRepository):
        self.repo = performance_repo

    def collect_from_backtest(
        self,
        strategy_name: str,
        statistics: Dict[str, Any],
        daily_returns: Optional[List[float]] = None,
    ) -> PerformanceSnapshot:
        """从回测统计结果采集绩效指标

        Args:
            strategy_name: 策略名称
            statistics: ConfigBacktestRunner._compute_statistics() 的输出
            daily_returns: 日收益率序列 (可选)

        Returns:
            PerformanceSnapshot 对象
        """
        returns_json = json.dumps(daily_returns) if daily_returns else None

        snapshot = PerformanceSnapshot(
            strategy_name=strategy_name,
            snapshot_date=date.today(),
            sharpe_ratio=statistics.get("sharpe_ratio"),
            sortino_ratio=statistics.get("sortino_ratio"),
            max_drawdown=statistics.get("max_drawdown"),
            annualized_return=statistics.get("annualized_return"),
            annualized_volatility=statistics.get("annualized_volatility"),
            win_rate=statistics.get("win_rate"),
            profit_factor=statistics.get("profit_factor"),
            total_trades=statistics.get("total_trades"),
            daily_returns=returns_json,
        )

        self.repo.save_snapshot(snapshot)
        return snapshot

    def collect_from_lazyframe(
        self,
        strategy_name: str,
        equity_curve: pl.LazyFrame,
    ) -> PerformanceSnapshot:
        """从权益曲线LazyFrame采集绩效指标

        Args:
            strategy_name: 策略名称
            equity_curve: 包含 date, equity 列的 LazyFrame

        Returns:
            PerformanceSnapshot 对象
        """
        df = equity_curve.collect()

        if "equity" not in df.columns or len(df) < 2:
            snapshot = PerformanceSnapshot(
                strategy_name=strategy_name,
                snapshot_date=date.today(),
            )
            self.repo.save_snapshot(snapshot)
            return snapshot

        equity = df["equity"].to_list()
        daily_returns = self._compute_daily_returns(equity)
        stats = self._compute_statistics(daily_returns)

        snapshot = PerformanceSnapshot(
            strategy_name=strategy_name,
            snapshot_date=date.today(),
            sharpe_ratio=stats.get("sharpe_ratio"),
            sortino_ratio=stats.get("sortino_ratio"),
            max_drawdown=stats.get("max_drawdown"),
            annualized_return=stats.get("annualized_return"),
            annualized_volatility=stats.get("annualized_volatility"),
            daily_returns=json.dumps(daily_returns),
        )

        self.repo.save_snapshot(snapshot)
        return snapshot

    def get_baseline_metrics(
        self, strategy_name: str, baseline_days: int = 252
    ) -> Optional[PerformanceSnapshot]:
        """获取基线指标 (用于漂移检测对比)"""
        return self.repo.get_baseline(strategy_name, baseline_days)

    @staticmethod
    def _compute_daily_returns(equity: List[float]) -> List[float]:
        """计算日收益率"""
        returns = []
        for i in range(1, len(equity)):
            if equity[i - 1] != 0:
                returns.append(equity[i] / equity[i - 1] - 1)
            else:
                returns.append(0.0)
        return returns

    @staticmethod
    def _compute_statistics(daily_returns: List[float]) -> Dict[str, Any]:
        """从日收益率计算统计指标"""
        import math

        if not daily_returns:
            return {}

        n = len(daily_returns)
        mean_r = sum(daily_returns) / n
        var_r = sum((r - mean_r) ** 2 for r in daily_returns) / max(n - 1, 1)
        std_r = math.sqrt(var_r)

        # 夏普比率 (rf=3%)
        rf_daily = 0.03 / 252
        sharpe = (mean_r - rf_daily) / std_r * math.sqrt(252) if std_r > 0 else 0.0

        # 索提诺比率
        downside = [r for r in daily_returns if r < rf_daily]
        downside_var = sum((r - rf_daily) ** 2 for r in downside) / max(len(downside), 1)
        downside_std = math.sqrt(downside_var)
        sortino = (mean_r - rf_daily) / downside_std * math.sqrt(252) if downside_std > 0 else 0.0

        # 最大回撤
        max_dd = 0.0
        cum = 1.0
        peak = 1.0
        for r in daily_returns:
            cum *= (1 + r)
            if cum > peak:
                peak = cum
            dd = (cum - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # 年化收益
        total_return = 1.0
        for r in daily_returns:
            total_return *= (1 + r)
        n_years = n / 252
        ann_return = total_return ** (1 / max(n_years, 0.01)) - 1 if total_return > 0 else -1.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(std_r * math.sqrt(252), 4),
        }
