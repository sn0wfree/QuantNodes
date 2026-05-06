# coding=utf-8
"""漂移检测器 - 三种检测算法"""

from __future__ import annotations

import math
from typing import List, Optional

from ..storage.models import DriftAlert, PerformanceSnapshot
from ..storage.repository import DriftAlertRepository, PerformanceRepository


class DriftDetector:
    """漂移检测器

    支持三种检测:
    1. KS检验: 收益率分布是否发生变化
    2. 夏普比率下降: 夏普比率是否显著下降
    3. 最大回撤超标: 最大回撤是否超过预设阈值
    """

    def __init__(
        self,
        alert_repo: DriftAlertRepository,
        perf_repo: PerformanceRepository,
        ks_threshold: float = 0.05,
        sharpe_drop_pct: float = 0.3,
        max_drawdown_limit: float = -0.20,
    ):
        self.alert_repo = alert_repo
        self.perf_repo = perf_repo
        self.ks_threshold = ks_threshold
        self.sharpe_drop_pct = sharpe_drop_pct
        self.max_drawdown_limit = max_drawdown_limit

    def detect_ks_drift(
        self,
        current_returns: List[float],
        baseline_returns: List[float],
        strategy_name: str,
    ) -> Optional[DriftAlert]:
        """KS检验: 收益率分布是否发生变化

        使用 scipy.stats.ks_2samp 检验两个样本是否来自同一分布。
        p_value < threshold → 分布发生显著变化 → 漂移。
        """
        if len(current_returns) < 5 or len(baseline_returns) < 5:
            return None

        try:
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(current_returns, baseline_returns)
        except ImportError:
            return None

        if p_value < self.ks_threshold:
            severity = "critical" if p_value < 0.01 else "warning"
            alert = DriftAlert(
                strategy_name=strategy_name,
                alert_type="ks_test",
                severity=severity,
                metric_name="return_distribution",
                current_value=statistic,
                p_value=p_value,
                message=(
                    f"收益分布发生漂移 (KS statistic={statistic:.4f}, "
                    f"p_value={p_value:.4f} < {self.ks_threshold})"
                ),
            )
            self.alert_repo.create_alert(alert)
            return alert
        return None

    def detect_sharpe_drop(
        self,
        current_snapshot: PerformanceSnapshot,
        baseline_snapshot: Optional[PerformanceSnapshot],
        strategy_name: str,
    ) -> Optional[DriftAlert]:
        """夏普比率下降检测

        当前夏普 < 基线夏普 * (1 - drop_pct) → 告警。
        """
        if baseline_snapshot is None or current_snapshot.sharpe_ratio is None:
            return None

        baseline_sharpe = baseline_snapshot.sharpe_ratio
        current_sharpe = current_snapshot.sharpe_ratio

        if baseline_sharpe is None or baseline_sharpe <= 0:
            return None

        threshold = baseline_sharpe * (1 - self.sharpe_drop_pct)
        if current_sharpe < threshold:
            drop_pct = (baseline_sharpe - current_sharpe) / baseline_sharpe
            severity = "critical" if drop_pct > 0.5 else "warning"
            alert = DriftAlert(
                strategy_name=strategy_name,
                alert_type="sharpe_drop",
                severity=severity,
                metric_name="sharpe_ratio",
                current_value=current_sharpe,
                baseline_value=baseline_sharpe,
                message=(
                    f"夏普比率下降 {drop_pct:.1%} "
                    f"(当前={current_sharpe:.4f}, 基线={baseline_sharpe:.4f}, "
                    f"阈值={threshold:.4f})"
                ),
            )
            self.alert_repo.create_alert(alert)
            return alert
        return None

    def detect_drawdown_breach(
        self,
        current_snapshot: PerformanceSnapshot,
        strategy_name: str,
    ) -> Optional[DriftAlert]:
        """最大回撤超标检测

        |current_dd| > max_allowed_dd → 告警。
        """
        if current_snapshot.max_drawdown is None:
            return None

        current_dd = current_snapshot.max_drawdown
        if abs(current_dd) > abs(self.max_drawdown_limit):
            severity = "critical" if abs(current_dd) > abs(self.max_drawdown_limit) * 1.5 else "warning"
            alert = DriftAlert(
                strategy_name=strategy_name,
                alert_type="drawdown_breach",
                severity=severity,
                metric_name="max_drawdown",
                current_value=current_dd,
                baseline_value=self.max_drawdown_limit,
                message=(
                    f"最大回撤超标 "
                    f"(当前={current_dd:.4f}, 限制={self.max_drawdown_limit:.4f})"
                ),
            )
            self.alert_repo.create_alert(alert)
            return alert
        return None

    def run_all_checks(
        self,
        strategy_name: str,
        current_returns: Optional[List[float]] = None,
    ) -> List[DriftAlert]:
        """运行所有漂移检测

        Args:
            strategy_name: 策略名称
            current_returns: 当前收益率序列 (用于KS检验)

        Returns:
            触发的告警列表
        """
        alerts = []

        # 获取最新快照和基线
        current = self.perf_repo.get_latest(strategy_name)
        baseline = self.perf_repo.get_baseline(strategy_name)

        if current is None:
            return alerts

        # 1. KS检验
        if current_returns and baseline:
            baseline_returns_str = baseline.daily_returns
            if baseline_returns_str:
                import json
                baseline_returns = json.loads(baseline_returns_str)
                alert = self.detect_ks_drift(current_returns, baseline_returns, strategy_name)
                if alert:
                    alerts.append(alert)

        # 2. 夏普比率下降
        alert = self.detect_sharpe_drop(current, baseline, strategy_name)
        if alert:
            alerts.append(alert)

        # 3. 最大回撤超标
        alert = self.detect_drawdown_breach(current, strategy_name)
        if alert:
            alerts.append(alert)

        return alerts
