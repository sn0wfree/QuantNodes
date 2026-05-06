# coding=utf-8
"""基于APScheduler的策略调度器"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def _run_strategy_job_static(strategy_name: str, config_path: str, **kwargs):
    """静态函数: 执行策略任务 (避免序列化问题)"""
    from .runner import StrategyRunner
    runner = StrategyRunner.from_default_db()
    runner.run_strategy(strategy_name, config_path, **kwargs)


class StrategyScheduler:
    """基于APScheduler的策略调度器

    支持三种触发方式:
    - cron: cron表达式定时
    - interval: 间隔定时
    - date: 指定时间执行一次
    """

    def __init__(self, db_path: str = "~/.quantnodes/monitor.db"):
        db_url = f"sqlite:///{Path(db_path).expanduser()}"
        self.scheduler = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=db_url)},
            executors={"default": ThreadPoolExecutor(max_workers=4)},
            job_defaults={"coalesce": True, "max_instances": 1},
        )
        self._started = False

    def add_cron_job(
        self,
        strategy_name: str,
        cron_expr: str,
        config_path: str,
        **kwargs,
    ) -> str:
        """添加cron定时任务

        Args:
            strategy_name: 策略名称 (作为job id)
            cron_expr: cron表达式, 如 "0 18 * * 1-5"
            config_path: YAML配置文件路径
            **kwargs: 传递给StrategyRunner.run_strategy的额外参数

        Returns:
            job id
        """
        job_id = f"strategy_{strategy_name}"
        parts = cron_expr.split()
        cron_kwargs = {}
        if len(parts) >= 5:
            cron_kwargs = {
                "minute": parts[0],
                "hour": parts[1],
                "day": parts[2],
                "month": parts[3],
                "day_of_week": parts[4],
            }

        self.scheduler.add_job(
            _run_strategy_job_static,
            "cron",
            id=job_id,
            replace_existing=True,
            args=[strategy_name, config_path],
            kwargs=kwargs,
            **cron_kwargs,
        )
        logger.info(f"Added cron job: {job_id} ({cron_expr})")
        return job_id

    def add_interval_job(
        self,
        strategy_name: str,
        interval_minutes: int,
        config_path: str,
        **kwargs,
    ) -> str:
        """添加间隔定时任务"""
        job_id = f"strategy_{strategy_name}"
        self.scheduler.add_job(
            _run_strategy_job_static,
            "interval",
            id=job_id,
            replace_existing=True,
            minutes=interval_minutes,
            args=[strategy_name, config_path],
            kwargs=kwargs,
        )
        logger.info(f"Added interval job: {job_id} ({interval_minutes}min)")
        return job_id

    def add_date_job(
        self,
        strategy_name: str,
        run_date,
        config_path: str,
        **kwargs,
    ) -> str:
        """添加一次性定时任务"""
        job_id = f"strategy_{strategy_name}_{run_date}"
        self.scheduler.add_job(
            _run_strategy_job_static,
            "date",
            id=job_id,
            run_date=run_date,
            args=[strategy_name, config_path],
            kwargs=kwargs,
        )
        logger.info(f"Added date job: {job_id} ({run_date})")
        return job_id

    def remove_job(self, strategy_name: str) -> bool:
        """移除任务"""
        job_id = f"strategy_{strategy_name}"
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception:
            return False

    def get_jobs(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = None
            try:
                next_run_time = job.next_run_time
                if next_run_time:
                    next_run = str(next_run_time)
            except Exception:
                pass
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": next_run,
                "trigger": str(job.trigger),
            })
        return jobs

    def pause_job(self, strategy_name: str) -> bool:
        """暂停任务"""
        job_id = f"strategy_{strategy_name}"
        try:
            self.scheduler.pause_job(job_id)
            return True
        except Exception:
            return False

    def resume_job(self, strategy_name: str) -> bool:
        """恢复任务"""
        job_id = f"strategy_{strategy_name}"
        try:
            self.scheduler.resume_job(job_id)
            return True
        except Exception:
            return False

    def start(self):
        """启动调度器"""
        if not self._started:
            self.scheduler.start()
            self._started = True
            logger.info("Scheduler started")

    def shutdown(self):
        """关闭调度器"""
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False
            logger.info("Scheduler shutdown")
