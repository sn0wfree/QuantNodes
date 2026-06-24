# coding=utf-8
"""
alpha_gpt_service.py - Alpha-GPT 工作流服务层

封装 AlphaGptWorkflow 提供：
- create_session: 异步启动工作流，返回 session_id
- get_status: 查询进度
- get_results: 获取结果
- stop: 停止会话
- list_sessions: 历史会话列表
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


class AlphaGptService:
    """Alpha-GPT 工作流服务（内存 session store）"""

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
                result = await asyncio.to_thread(workflow.run)

                session.result = {
                    "objective": result.objective,
                    "iterations_completed": result.iterations_completed,
                    "total_formulas": result.total_formulas,
                    "final_pool": [f.to_dict() for f in result.final_pool],
                    "summary": result.summary,
                    "elapsed_seconds": result.elapsed_seconds,
                }
                session.status = "completed"
            except asyncio.CancelledError:
                session.status = "stopped"
                raise
            except Exception as exc:
                logger.exception("Alpha-GPT session %s failed", session.session_id)
                session.error = str(exc)
                session.status = "failed"
            finally:
                session.completed_at = time.time()
                if session.started_at:
                    session.elapsed_seconds = session.completed_at - session.started_at
                self._tasks.pop(session.session_id, None)

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
