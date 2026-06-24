# coding=utf-8
"""
alpha_gpt_service.py - Alpha-GPT 工作流服务层

封装 AlphaGptWorkflow 提供：
- create_session: 异步启动工作流，返回 session_id
- get_status: 查询进度
- get_results: 获取结果
- stop: 停止会话
- list_sessions: 历史会话列表
- subscribe_events: 订阅 WebSocket 流式事件（v2.7.0+）
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

ALPHA_GPT_EVENT_TYPES = {
    "round_started",
    "subagent_started",
    "subagent_done",
    "formulas_evaluated",
    "round_completed",
    "final_pool_ready",
    "error",
    "done",
}


@dataclass
class AlphaGptSession:
    """Alpha-GPT 工作流会话状态"""

    session_id: str
    objective: str
    config_dict: Dict[str, Any]
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


class AlphaGptService:
    """Alpha-GPT 工作流服务（内存 session store + 事件总线）"""

    def __init__(self, max_concurrent: int = 3):
        self._sessions: Dict[str, AlphaGptSession] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "objective": s.objective,
                "status": s.status,
                "created_at": s.created_at,
                "completed_at": s.completed_at,
                "elapsed_seconds": s.elapsed_seconds,
            }
            for s in self._sessions.values()
        ]

    def get_session(self, session_id: str) -> Optional[AlphaGptSession]:
        return self._sessions.get(session_id)

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self.get_session(session_id)
        if s is None:
            return None
        return {
            "session_id": s.session_id,
            "status": s.status,
            "objective": s.objective,
            "created_at": s.created_at,
            "started_at": s.started_at,
            "completed_at": s.completed_at,
            "elapsed_seconds": s.elapsed_seconds,
            "progress": s.progress,
        }

    def get_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self.get_session(session_id)
        if s is None:
            return None
        if s.status != "completed":
            return {"session_id": s.session_id, "status": s.status, "result": None}
        return s.result

    def stop_session(self, session_id: str) -> bool:
        s = self.get_session(session_id)
        if s is None:
            return False
        if s.status in {"completed", "failed", "stopped"}:
            return False
        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        s.status = "stopped"
        s.completed_at = time.time()
        return True

    async def create_session(
        self,
        objective: str,
        data_path: Optional[str] = None,
        iterations: int = 5,
        pool_size: int = 10,
        top_k: int = 10,
        forward_returns: Optional[List[int]] = None,
        llm_provider: str = "mock",
        enable_backtest: bool = False,
        **kwargs: Any,
    ) -> str:
        """启动新会话（异步）"""
        session_id = f"agpt-{uuid.uuid4().hex[:12]}"
        session = AlphaGptSession(
            session_id=session_id,
            objective=objective,
            config_dict={
                "data_path": data_path,
                "iterations": iterations,
                "pool_size": pool_size,
                "top_k": top_k,
                "forward_returns": forward_returns or [1, 5, 20],
                "llm_provider": llm_provider,
                "enable_backtest": enable_backtest,
                **kwargs,
            },
        )
        self._sessions[session_id] = session

        task = asyncio.create_task(self._run_session(session))
        self._tasks[session_id] = task

        return session_id

    async def _run_session(self, session: AlphaGptSession) -> None:
        """执行工作流（后台 task）"""
        async with self._semaphore:
            session.status = "running"
            session.started_at = time.time()
            try:
                cfg = session.config_dict
                df = await asyncio.to_thread(
                    self._load_data, cfg.get("data_path"),
                )

                from QuantNodes.research.quant_alpha.workflow import (
                    AlphaGptConfig, AlphaGptWorkflow,
                )

                config = AlphaGptConfig(
                    objective=session.objective,
                    iterations=cfg["iterations"],
                    pool_size=cfg["pool_size"],
                    top_k=cfg["top_k"],
                    forward_returns=cfg["forward_returns"],
                    llm_provider=cfg["llm_provider"],
                    enable_backtest=cfg.get("enable_backtest", False),
                )
                workflow = AlphaGptWorkflow(config=config, data=df)

                await self._emit(session, {
                    "type": "round_started",
                    "total_rounds": cfg["iterations"],
                    "round": 0,
                })

                # 在 outer async 上下文中捕获 running loop，传给 to_thread
                main_loop = asyncio.get_running_loop()
                result = await asyncio.to_thread(
                    self._run_workflow_with_events, workflow, session, main_loop,
                )

                session.result = {
                    "objective": result.objective,
                    "iterations_completed": result.iterations_completed,
                    "total_formulas": result.total_formulas,
                    "final_pool": [f.to_dict() for f in result.final_pool],
                    "summary": result.summary,
                    "elapsed_seconds": result.elapsed_seconds,
                }

                await self._emit(session, {
                    "type": "final_pool_ready",
                    "pool": [f.to_dict() for f in result.final_pool],
                    "summary": result.summary,
                })
                await self._emit(session, {"type": "done"})
                session.status = "completed"
            except asyncio.CancelledError:
                session.status = "stopped"
                await self._emit(session, {"type": "error", "message": "stopped"})
                raise
            except Exception as exc:
                logger.exception("Alpha-GPT session %s failed", session.session_id)
                session.error = str(exc)
                session.status = "failed"
                await self._emit(session, {"type": "error", "message": str(exc)})
            finally:
                session.completed_at = time.time()
                if session.started_at:
                    session.elapsed_seconds = session.completed_at - session.started_at
                self._tasks.pop(session.session_id, None)

    def _run_workflow_with_events(
        self,
        workflow: Any,
        session: AlphaGptSession,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Any:
        """同步跑 workflow，期间通过 main_loop 异步发事件"""
        from QuantNodes.research.quant_alpha.workflow import (
            AlphaGptWorkflow,
        )

        # 简化：直接调用 workflow.run()，每个 round 后 emit round_completed
        original_run_one_round = AlphaGptWorkflow._run_one_round

        def _schedule(coro: Any) -> None:
            if main_loop is not None:
                asyncio.run_coroutine_threadsafe(coro, main_loop)

        def patched_run_one_round(self: Any, round_idx: int) -> None:
            _schedule(self._service_emit(session, {
                "type": "round_started",
                "round": round_idx,
                "total_rounds": self.config.iterations,
            }))
            original_run_one_round(self, round_idx)
            evals = self.state.all_evaluations
            recent: List[Dict[str, Any]] = []
            if evals:
                try:
                    recent = [
                        e.to_dict() for e in evals
                        if int(e.formula_id.split("-")[1]) == round_idx
                    ]
                    if not recent:
                        recent = [e.to_dict() for e in evals[-self.config.pool_size:]]
                except Exception:
                    recent = [e.to_dict() for e in evals[-self.config.pool_size:]]
            _schedule(self._service_emit(session, {
                "type": "round_completed",
                "round": round_idx,
                "total_rounds": self.config.iterations,
                "formulas_evaluated": len(recent),
                "best_ir": max(
                    (e["ir"] for e in recent if e["status"] == "success"),
                    default=0.0,
                ),
            }))

        AlphaGptWorkflow._service_emit = self._emit  # type: ignore[attr-defined]
        AlphaGptWorkflow._run_one_round = patched_run_one_round  # type: ignore[method-assign]
        try:
            return workflow.run()
        finally:
            AlphaGptWorkflow._run_one_round = original_run_one_round  # type: ignore[method-assign]
            if hasattr(AlphaGptWorkflow, "_service_emit"):
                delattr(AlphaGptWorkflow, "_service_emit")

    async def _emit(self, session: AlphaGptSession, event: Dict[str, Any]) -> None:
        """异步发事件到所有订阅者（WebSocket + 历史 buffer）"""
        event.setdefault("session_id", session.session_id)
        event.setdefault("ts", time.time())
        session.events.append(event)
        dead = set()
        for queue in session.subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
        session.subscribers -= dead

    def subscribe(self, session_id: str) -> Optional[asyncio.Queue]:
        """订阅一个 session 的事件流（返回 queue，需要用 unsubscribe 清理）"""
        s = self.get_session(session_id)
        if s is None:
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        s.subscribers.add(queue)
        for evt in s.events:
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:
                break
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        s = self.get_session(session_id)
        if s is not None:
            s.subscribers.discard(queue)

    @staticmethod
    def _load_data(data_path: Optional[str]) -> Any:
        """同步加载数据"""
        if data_path is None:
            import polars as pl
            import numpy as np
            np.random.seed(42)
            dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
            rows = []
            for date in dates:
                for code in ["A", "B", "C", "D", "E"]:
                    close = float(np.random.randn() * 5 + 100)
                    rows.append({
                        "date": date, "code": code,
                        "close": close, "open": close,
                        "high": close + 1.0, "low": close - 1.0,
                        "vol": 1000.0,
                    })
            return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())

        p = Path(data_path)
        import polars as pl
        if p.suffix == ".parquet":
            return pl.read_parquet(p)
        if p.suffix == ".csv":
            return pl.read_csv(p)
        raise ValueError(f"Unsupported data format: {p.suffix}")


# 全局单例
alpha_gpt_service = AlphaGptService()


__all__ = ["AlphaGptService", "AlphaGptSession", "alpha_gpt_service"]
