# coding=utf-8
"""
mine_logics router - 自动化因子挖掘 REST API + WebSocket (v3.0.3)

REST:
- POST /api/mine-logics/start           启动批量挖掘
- GET  /api/mine-logics/status/{run_id} 查询进度
- GET  /api/mine-logics/results/{run_id} 获取结果
- POST /api/mine-logics/stop/{run_id}   停止运行
- GET  /api/mine-logics/history         历史运行列表

WebSocket:
- WS /api/mine-logics/stream/{run_id}  实时事件流
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..services.mine_logics_service import mine_logics_service

router = APIRouter()
logger = logging.getLogger(__name__)


# ==============================================================================
# Request / Response models
# ==============================================================================

class MineLogicsStartRequest(BaseModel):
    source_libs: List[str] = Field(
        default=["alpha101", "alpha158", "alpha191"],
        description="来源库列表",
    )
    max_per_lib: int = Field(10, ge=1, le=100, description="每个库最多挖掘条数")
    workers: int = Field(4, ge=1, le=16, description="并发线程数")
    wiki_path: str = Field("wiki_auto", description="Wiki 根目录")
    live: bool = Field(False, description="使用真实 LLM")
    strict: bool = Field(False, description="严格模式")
    skip_existing: bool = Field(True, description="跳过已存在 Logic pages")


# ==============================================================================
# REST Endpoints
# ==============================================================================

@router.post("/start")
async def start(req: MineLogicsStartRequest) -> Dict[str, Any]:
    """启动批量因子挖掘 (async)"""
    run_id = await mine_logics_service.start(
        source_libs=req.source_libs,
        max_per_lib=req.max_per_lib,
        workers=req.workers,
        wiki_path=req.wiki_path,
        live=req.live,
        strict=req.strict,
        skip_existing=req.skip_existing,
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("/status/{run_id}")
async def status(run_id: str) -> Dict[str, Any]:
    """查询运行进度"""
    s = mine_logics_service.get_status(run_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return s


@router.get("/results/{run_id}")
async def results(run_id: str) -> Dict[str, Any]:
    """获取运行结果"""
    r = mine_logics_service.get_results(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return r


@router.post("/stop/{run_id}")
async def stop(run_id: str) -> Dict[str, Any]:
    """停止运行"""
    ok = mine_logics_service.stop(run_id)
    if not ok:
        run = mine_logics_service.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return {"stopped": False, "reason": f"run already {run.status}"}
    return {"stopped": True}


@router.get("/history")
async def history() -> Dict[str, Any]:
    """历史运行列表"""
    runs = mine_logics_service.list_history()
    return {"runs": runs, "total": len(runs)}


# ==============================================================================
# WebSocket: 实时事件流
# ==============================================================================

@router.websocket("/stream/{run_id}")
async def stream(websocket: WebSocket, run_id: str) -> None:
    """实时事件流 (WebSocket)

    协议:
    - Client opens: ws://host/api/mine-logics/stream/{run_id}
    - Server 发送 buffered events (含历史)
    - Server 持续推送新 events 直到 run 完成
    - Server 发送 {"type": "done"} 后关闭连接

    Events:
    - mining_started:     批量挖掘开始
    - formula_attempted:  一个 formula 开始 (formula_id, done, total)
    - formula_completed:  一个 formula 完成 (formula_id, success, parse_layer)
    - batch_completed:    全部完成 (n_mined, n_skipped, n_failed)
    - error:              致命错误
    - done:               会话结束
    - heartbeat:          30s 超时心跳
    """
    await websocket.accept()
    run = mine_logics_service.get_run(run_id)
    if run is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"run not found: {run_id}",
        }))
        await websocket.close()
        return

    queue = mine_logics_service.subscribe(run_id)
    if queue is None:
        await websocket.close()
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "ts": asyncio.get_event_loop().time(),
                }))
                continue

            await websocket.send_text(json.dumps(event, default=str))

            if event.get("type") in {"done", "error"}:
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket stream error")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
        except Exception:
            pass
    finally:
        mine_logics_service.unsubscribe(run_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
