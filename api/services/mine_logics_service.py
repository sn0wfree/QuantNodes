# coding=utf-8
"""
mine_logics_service.py - 自动化因子挖掘服务层 (v3.0.3)

封装 mine_logic_library_v2 提供:
- start: 异步启动批量挖掘，返回 run_id
- get_status: 查询进度
- get_results: 获取结果
- stop: 停止运行
- list_history: 历史运行列表 (JSON 持久化)
- subscribe_events: 订阅 WebSocket 流式事件

与 alpha_gpt_service.py 同构: 内存 session store + asyncio.Queue 事件总线
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ==============================================================================
# Event types (WebSocket 流式协议)
# ==============================================================================

MINE_LOGICS_EVENT_TYPES = {
    "mining_started",
    "formula_attempted",
    "formula_completed",
    "batch_completed",
    "error",
    "done",
}

HISTORY_PATH = Path("data/mine_runs/history.json")


# ==============================================================================
# Session dataclass
# ==============================================================================

@dataclass
class MineLogicsRun:
    """一次批量挖掘运行的状态"""

    run_id: str
    config: Dict[str, Any]
    status: str = "pending"  # pending | running | completed | failed | stopped
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_seconds: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)


# ==============================================================================
# Service
# ==============================================================================

class MineLogicsService:
    """自动化因子挖掘服务 (内存 session store + 事件总线 + JSON history)"""

    def __init__(self, max_concurrent: int = 3):
        self._runs: Dict[str, MineLogicsRun] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._history_path = HISTORY_PATH
        self._load_history()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[MineLogicsRun]:
        return self._runs.get(run_id)

    def get_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_run(run_id)
        if run is None:
            return None
        return {
            "run_id": run.run_id,
            "status": run.status,
            "config": run.config,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "elapsed_seconds": run.elapsed_seconds,
            "progress": run.progress,
        }

    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_run(run_id)
        if run is None:
            return None
        if run.status != "completed":
            return {"run_id": run.run_id, "status": run.status, "result": None}
        return {
            "run_id": run.run_id,
            "status": run.status,
            "result": run.result,
        }

    def stop(self, run_id: str) -> bool:
        run = self.get_run(run_id)
        if run is None:
            return False
        if run.status in {"completed", "failed", "stopped"}:
            return False
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        run.status = "stopped"
        run.completed_at = time.time()
        run.elapsed_seconds = run.completed_at - (run.started_at or run.created_at)
        return True

    def list_history(self) -> List[Dict[str, Any]]:
        """返回历史运行列表 (合并内存 + JSON 持久化)"""
        history = []
        # JSON 持久化的
        for entry in self._history_list:
            history.append(entry)
        # 内存中的 (可能更新)
        seen_ids = {h["run_id"] for h in history}
        for run in self._runs.values():
            if run.run_id not in seen_ids:
                history.append(self._run_to_summary(run))
            else:
                # 用内存中的最新状态覆盖
                for i, h in enumerate(history):
                    if h["run_id"] == run.run_id:
                        history[i] = self._run_to_summary(run)
                        break
        history.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return history

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(
        self,
        source_libs: Optional[List[str]] = None,
        max_per_lib: int = 10,
        workers: int = 4,
        wiki_path: str = "wiki_auto",
        live: bool = False,
        strict: bool = False,
        skip_existing: bool = True,
    ) -> str:
        """启动新运行 (异步)"""
        run_id = f"ml-{uuid.uuid4().hex[:12]}"
        config = {
            "source_libs": source_libs or ["alpha101", "alpha158", "alpha191"],
            "max_per_lib": max_per_lib,
            "workers": workers,
            "wiki_path": wiki_path,
            "live": live,
            "strict": strict,
            "skip_existing": skip_existing,
        }
        run = MineLogicsRun(run_id=run_id, config=config)
        self._runs[run_id] = run

        task = asyncio.create_task(self._run_session(run))
        self._tasks[run_id] = task

        return run_id

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------

    async def _run_session(self, run: MineLogicsRun) -> None:
        """执行批量挖掘 (后台 task)"""
        async with self._semaphore:
            run.status = "running"
            run.started_at = time.time()
            loop = asyncio.get_event_loop()

            try:
                cfg = run.config

                # 构建 LLM 客户端
                llm_client = None
                if cfg.get("live"):
                    try:
                        from QuantNodes.ai.llm.gateway import get_llm_gateway
                        llm_client = get_llm_gateway()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to initialize LLM gateway: {exc}"
                        ) from exc

                # 构建 strict
                strict = None
                if cfg.get("strict"):
                    from QuantNodes.research.quant_alpha.logic_mining.metrics import (
                        StrictConfig,
                    )
                    strict = StrictConfig(call=True, parse=True, structured=True)

                # 进度回调: emit 事件到 WebSocket
                def _on_progress(done: int, total: int, fid: str) -> None:
                    event = {
                        "type": "formula_attempted",
                        "formula_id": fid,
                        "done": done,
                        "total": total,
                    }
                    asyncio.run_coroutine_threadsafe(
                        self._emit(run.run_id, event), loop
                    )

                # 发 mining_started 事件
                await self._emit(run.run_id, {
                    "type": "mining_started",
                    "run_id": run.run_id,
                    "source_libs": cfg["source_libs"],
                })

                # 在线程池中运行同步 batch engine
                from QuantNodes.research.quant_alpha.logic_mining.batch import (
                    ThreadSafeMetrics,
                    mine_logic_library_v2,
                )

                metrics = ThreadSafeMetrics()
                batch = await asyncio.to_thread(
                    mine_logic_library_v2,
                    source_libs=cfg["source_libs"],
                    llm_client=llm_client,
                    max_per_lib=cfg["max_per_lib"],
                    workers=cfg["workers"],
                    metrics=metrics,
                    strict=strict,
                    wiki_path=cfg["wiki_path"],
                    skip_existing=cfg.get("skip_existing", True),
                    on_progress=_on_progress,
                )

                # 构建结果
                from QuantNodes.research.quant_alpha.logic_mining.report import (
                    MetricsReportBuilder,
                )
                report = MetricsReportBuilder.from_batch(batch)

                run.result = {
                    "summary": report.to_dict()["summary"],
                    "source_breakdown": report.source_lib_breakdown,
                    "top_factors": [
                        {
                            "formula_id": e.formula_id,
                            "formula": e.formula,
                            "source_lib": e.source_lib,
                            "ir": e.ir,
                            "ic_mean": e.ic_mean,
                            "rank_ic": e.rank_ic,
                            "parse_layer": 0,
                            "tags": e.tags,
                        }
                        for e in (batch.pool.select(top_n=50) if batch.pool else [])
                    ],
                    "agent_stats": report.agent_stats,
                    "warnings": report.warnings,
                    "wall_clock_s": batch.wall_clock_s,
                }

                # 发 batch_completed 事件
                await self._emit(run.run_id, {
                    "type": "batch_completed",
                    "n_mined": batch.n_mined,
                    "n_skipped": batch.n_skipped,
                    "n_failed": batch.n_failed,
                    "wall_clock_s": batch.wall_clock_s,
                })

                run.status = "completed"
                run.completed_at = time.time()
                run.elapsed_seconds = run.completed_at - run.started_at

                # 写入历史
                self._save_history(run)

            except asyncio.CancelledError:
                run.status = "stopped"
                run.completed_at = time.time()
                run.elapsed_seconds = run.completed_at - run.started_at
            except Exception as exc:
                logger.exception("MineLogics run failed: %s", exc)
                run.status = "failed"
                run.error = repr(exc)
                run.completed_at = time.time()
                run.elapsed_seconds = run.completed_at - run.started_at
                await self._emit(run.run_id, {
                    "type": "error",
                    "message": str(exc),
                })
            finally:
                # 发 done 事件
                await self._emit(run.run_id, {"type": "done"})

    # ------------------------------------------------------------------
    # Event bus (subscribe / unsubscribe / emit)
    # ------------------------------------------------------------------

    def subscribe(self, run_id: str) -> Optional[asyncio.Queue]:
        """订阅一个 run 的事件流 (返回 queue)"""
        run = self.get_run(run_id)
        if run is None:
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        run.subscribers.add(queue)
        # 回放历史 events
        for evt in run.events:
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:
                break
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        run = self.get_run(run_id)
        if run is not None:
            run.subscribers.discard(queue)

    async def _emit(self, run_id: str, event: Dict[str, Any]) -> None:
        """异步发事件到所有订阅者 + 历史 buffer"""
        run = self.get_run(run_id)
        if run is None:
            return
        event.setdefault("run_id", run_id)
        event.setdefault("ts", time.time())
        run.events.append(event)
        # 更新 progress
        self._update_progress(run, event)
        # 广播到 subscribers
        dead = set()
        for queue in run.subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
        run.subscribers -= dead

    def _update_progress(self, run: MineLogicsRun, event: Dict[str, Any]) -> None:
        """根据事件类型更新 progress 字典"""
        etype = event.get("type")
        if etype == "mining_started":
            run.progress["source_libs"] = event.get("source_libs", [])
        elif etype == "formula_attempted":
            run.progress["current_formula"] = event.get("formula_id")
            run.progress["done"] = event.get("done", 0)
            run.progress["total"] = event.get("total", 0)
        elif etype == "batch_completed":
            run.progress["n_mined"] = event.get("n_mined", 0)
            run.progress["n_skipped"] = event.get("n_skipped", 0)
            run.progress["n_failed"] = event.get("n_failed", 0)

    # ------------------------------------------------------------------
    # History persistence (JSON)
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        """从 JSON 文件加载历史"""
        self._history_list: List[Dict[str, Any]] = []
        if self._history_path.exists():
            try:
                data = json.loads(self._history_path.read_text(encoding="utf-8"))
                self._history_list = data if isinstance(data, list) else []
            except Exception as exc:
                logger.warning("Failed to load history: %s", exc)

    def _save_history(self, run: MineLogicsRun) -> None:
        """把 run 写入 JSON 历史文件"""
        entry = self._run_to_summary(run)
        # 更新或追加
        for i, h in enumerate(self._history_list):
            if h["run_id"] == run.run_id:
                self._history_list[i] = entry
                break
        else:
            self._history_list.append(entry)
        # 保持最新 100 条
        self._history_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        self._history_list = self._history_list[:100]
        # 写文件
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(
                json.dumps(self._history_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save history: %s", exc)

    @staticmethod
    def _run_to_summary(run: MineLogicsRun) -> Dict[str, Any]:
        """run → JSON 可序列化的 summary dict"""
        return {
            "run_id": run.run_id,
            "status": run.status,
            "config": run.config,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "elapsed_seconds": run.elapsed_seconds,
            "progress": dict(run.progress),
            "error": run.error,
        }


# 全局单例
mine_logics_service = MineLogicsService()
