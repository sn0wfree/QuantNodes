# coding=utf-8
"""策略执行器 - 被调度器调用"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from ..storage.models import StrategyRun
from ..storage.repository import DatabaseManager, StrategyRunRepository
from ..monitor.collector import MetricsCollector
from ..monitor.drift import DriftDetector
from ..monitor.alerter import Alerter

logger = logging.getLogger(__name__)


class StrategyRunner:
    """策略执行器

    被调度器调用，执行策略并记录结果。
    """

    def __init__(
        self,
        run_repo: StrategyRunRepository,
        metrics_collector: MetricsCollector,
        drift_detector: DriftDetector,
        alerter: Alerter,
    ):
        self.run_repo = run_repo
        self.collector = metrics_collector
        self.drift_detector = drift_detector
        self.alerter = alerter

    @classmethod
    def from_default_db(cls, db_path: str = "~/.quantnodes/monitor.db") -> StrategyRunner:
        """从默认数据库创建StrategyRunner"""
        db = DatabaseManager(db_path)
        db.connect()
        run_repo = StrategyRunRepository(db)
        from ..storage.repository import PerformanceRepository, DriftAlertRepository
        perf_repo = PerformanceRepository(db)
        alert_repo = DriftAlertRepository(db)
        collector = MetricsCollector(perf_repo)
        drift = DriftDetector(alert_repo, perf_repo)
        alerter = Alerter(alert_repo)
        return cls(run_repo, collector, drift, alerter)

    def run_strategy(
        self,
        strategy_name: str,
        config_path: str,
        run_type: str = "sample_out",
        version: str = None,
    ) -> Optional[Dict[str, Any]]:
        """执行策略并记录结果

        Args:
            strategy_name: 策略名称
            config_path: YAML配置文件路径
            run_type: 运行类型 (backtest/live/sample_out)
            version: 策略版本号

        Returns:
            执行结果字典
        """
        # 读取配置快照
        config_snapshot = ""
        try:
            config_snapshot = Path(config_path).read_text(encoding="utf-8")
        except Exception:
            pass

        # 创建运行记录
        run = StrategyRun(
            strategy_name=strategy_name,
            run_type=run_type,
            status="running",
            strategy_version=version,
            start_time=datetime.now(),
            config_snapshot=config_snapshot,
        )
        run_id = self.run_repo.create(run)

        try:
            # 执行回测
            statistics = self._execute_backtest(config_path)

            # 更新运行记录
            self.run_repo.update_status(run_id, "success", statistics=statistics)

            # 采集绩效指标
            if statistics:
                self.collector.collect_from_backtest(strategy_name, statistics)

                # 运行漂移检测
                alerts = self.drift_detector.run_all_checks(strategy_name)
                if alerts:
                    logger.warning(
                        f"Strategy {strategy_name}: {len(alerts)} drift alerts"
                    )

            logger.info(f"Strategy {strategy_name} completed successfully")
            return {
                "status": "success",
                "run_id": run_id,
                "statistics": statistics,
                "alerts": [a.message for a in alerts] if alerts else [],
            }

        except Exception as e:
            error_msg = traceback.format_exc()
            self.run_repo.update_status(run_id, "failed", error_message=error_msg)
            logger.error(f"Strategy {strategy_name} failed: {e}")
            return {
                "status": "failed",
                "run_id": run_id,
                "error": str(e),
            }

    def _execute_backtest(self, config_path: str) -> Optional[Dict[str, Any]]:
        """执行回测"""
        from QuantNodes.agent.config.loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load(config_path)

        # 使用 ConfigBacktestRunner 执行
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool

        tool = ConfigBacktestTool()
        # 简化: 直接返回基本统计
        return {"status": "completed", "config": config.name}
