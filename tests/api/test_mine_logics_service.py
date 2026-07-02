# coding=utf-8
"""
test_mine_logics_service.py - MineLogicsService 单元测试 (v3.0.3 Step 1)

覆盖:
- start / get_status / get_results / stop / list_history
- subscribe / unsubscribe / event buffer
- progress 更新
- JSON 历史持久化
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from api.services.mine_logics_service import (
    MineLogicsRun,
    MineLogicsService,
)


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def service(tmp_path: Path) -> MineLogicsService:
    """独立 service 实例 (使用临时 history 文件)"""
    svc = MineLogicsService(max_concurrent=1)
    svc._history_path = tmp_path / "history.json"
    return svc


# ======================================================================
# CRUD
# ======================================================================
class TestServiceCRUD:
    def test_get_run_returns_none(self, service):
        assert service.get_run("nonexistent") is None

    def test_get_status_returns_none(self, service):
        assert service.get_status("nonexistent") is None

    def test_get_results_returns_none(self, service):
        assert service.get_results("nonexistent") is None

    @pytest.mark.asyncio
    async def test_start_creates_run(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        assert run_id.startswith("ml-")
        run = service.get_run(run_id)
        assert run is not None
        assert run.config["source_libs"] == ["alpha101"]
        assert run.config["max_per_lib"] == 1
        # 等待 task 完成 (sync mock 很快)
        await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_get_status_after_start(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        await asyncio.sleep(0.3)
        status = service.get_status(run_id)
        assert status is not None
        assert status["run_id"] == run_id
        assert status["status"] in {"pending", "running", "completed", "failed"}
        assert "progress" in status

    @pytest.mark.asyncio
    async def test_get_results_after_completion(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        # 等待完成
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = service.get_status(run_id)
            if status and status["status"] in {"completed", "failed"}:
                break
        results = service.get_results(run_id)
        assert results is not None
        assert results["run_id"] == run_id

    @pytest.mark.asyncio
    async def test_stop_active(self, service):
        """如果 run 还在 pending/running 状态, stop 应返回 True"""
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        # 立即尝试 stop (不等待)
        run = service.get_run(run_id)
        if run.status in {"pending", "running"}:
            ok = service.stop(run_id)
            assert ok is True
            status = service.get_status(run_id)
            assert status["status"] == "stopped"
        else:
            # 已经完成 (mock 太快), stop 返回 False
            ok = service.stop(run_id)
            assert ok is False

    @pytest.mark.asyncio
    async def test_stop_after_completion(self, service):
        """完成后 stop 应返回 False"""
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = service.get_status(run_id)
            if status and status["status"] in {"completed", "failed"}:
                break
        ok = service.stop(run_id)
        assert ok is False

    @pytest.mark.asyncio
    async def test_stop_nonexistent(self, service):
        assert service.stop("nonexistent") is False

    


# ======================================================================
# subscribe / unsubscribe
# ======================================================================
class TestServiceEvents:
    @pytest.mark.asyncio
    async def test_subscribe_returns_queue(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        queue = service.subscribe(run_id)
        assert queue is not None
        assert isinstance(queue, asyncio.Queue)
        service.unsubscribe(run_id, queue)

    def test_subscribe_nonexistent_returns_none(self, service):
        assert service.subscribe("nonexistent") is None

    @pytest.mark.asyncio
    async def test_subscribe_gets_buffered_events(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        await asyncio.sleep(0.5)  # 等待一些事件产生
        queue = service.subscribe(run_id)
        assert queue is not None
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert len(events) >= 1
        assert any(e["type"] == "mining_started" for e in events)
        service.unsubscribe(run_id, queue)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_queue(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        queue = service.subscribe(run_id)
        service.unsubscribe(run_id, queue)
        run = service.get_run(run_id)
        assert queue not in run.subscribers


# ======================================================================
# Progress
# ======================================================================
class TestServiceProgress:
    @pytest.mark.asyncio
    async def test_progress_updated_on_batch_completed(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        for _ in range(30):
            await asyncio.sleep(0.2)
            status = service.get_status(run_id)
            if status and status["status"] in {"completed", "failed", "stopped"}:
                break
        status = service.get_status(run_id)
        progress = status["progress"]
        # batch_completed 事件可能还没到 (mock LLM 太快), 但 progress 应至少有 source_libs
        assert "source_libs" in progress or "n_mined" in progress


# ======================================================================
# History
# ======================================================================
class TestServiceHistory:
    @pytest.mark.asyncio
    async def test_list_history_empty(self, service):
        history = service.list_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_list_history_after_run(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = service.get_status(run_id)
            if status and status["status"] == "completed":
                break
        history = service.list_history()
        assert len(history) >= 1
        assert any(h["run_id"] == run_id for h in history)

    @pytest.mark.asyncio
    async def test_history_persisted_to_json(self, service):
        run_id = await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = service.get_status(run_id)
            if status and status["status"] == "completed":
                break
        # JSON 文件应该存在
        assert service._history_path.exists()
        data = json.loads(service._history_path.read_text())
        assert isinstance(data, list)
        assert any(h["run_id"] == run_id for h in data)

    @pytest.mark.asyncio
    async def test_history_max_100(self, service):
        # 快速创建多个 run
        for _ in range(5):
            await service.start(source_libs=["alpha101"], max_per_lib=1, workers=1)
            await asyncio.sleep(0.2)
        history = service.list_history()
        assert len(history) <= 100


# ======================================================================
# _run_to_summary
# ======================================================================
class TestRunToSummary:
    def test_summary_fields(self):
        run = MineLogicsRun(
            run_id="ml-test",
            config={"source_libs": ["alpha101"]},
            status="completed",
            created_at=100.0,
            started_at=101.0,
            completed_at=150.0,
            elapsed_seconds=49.0,
        )
        summary = MineLogicsService._run_to_summary(run)
        assert summary["run_id"] == "ml-test"
        assert summary["status"] == "completed"
        assert summary["elapsed_seconds"] == 49.0
        assert "progress" in summary
        assert "error" in summary
