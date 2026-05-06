# coding=utf-8
"""调度器测试"""

import pytest
import tempfile
import os

from QuantNodes.monitor.scheduler.scheduler import StrategyScheduler


class TestStrategyScheduler:

    def test_add_cron_job(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            scheduler = StrategyScheduler(db_path)
            scheduler.start()
            job_id = scheduler.add_cron_job(
                "test_strategy", "0 18 * * 1-5", "/tmp/test.yaml",
            )
            assert job_id == "strategy_test_strategy"
            jobs = scheduler.get_jobs()
            assert len(jobs) == 1
            scheduler.shutdown()
        finally:
            os.unlink(db_path)

    def test_add_interval_job(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            scheduler = StrategyScheduler(db_path)
            scheduler.start()
            job_id = scheduler.add_interval_job(
                "test_strategy", 60, "/tmp/test.yaml",
            )
            assert job_id == "strategy_test_strategy"
            scheduler.shutdown()
        finally:
            os.unlink(db_path)

    def test_remove_job(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            scheduler = StrategyScheduler(db_path)
            scheduler.start()
            scheduler.add_cron_job("s1", "0 18 * * 1-5", "/tmp/test.yaml")
            assert scheduler.remove_job("s1") is True
            assert scheduler.remove_job("nonexistent") is False
            scheduler.shutdown()
        finally:
            os.unlink(db_path)

    def test_list_jobs(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            scheduler = StrategyScheduler(db_path)
            scheduler.start()
            scheduler.add_cron_job("s1", "0 18 * * 1-5", "/tmp/test.yaml")
            scheduler.add_interval_job("s2", 60, "/tmp/test.yaml")
            jobs = scheduler.get_jobs()
            assert len(jobs) == 2
            scheduler.shutdown()
        finally:
            os.unlink(db_path)

    def test_pause_resume(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            scheduler = StrategyScheduler(db_path)
            scheduler.start()
            scheduler.add_cron_job("s1", "0 18 * * 1-5", "/tmp/test.yaml")
            assert scheduler.pause_job("s1") is True
            assert scheduler.resume_job("s1") is True
            scheduler.shutdown()
        finally:
            os.unlink(db_path)
