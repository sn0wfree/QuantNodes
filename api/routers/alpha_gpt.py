# coding=utf-8
"""
alpha_gpt router - Alpha-GPT 6 端点 REST API + 1 WebSocket

REST:
- POST /api/alpha/alpha-gpt/generate  启动工作流
- GET  /api/alpha/alpha-gpt/status/{sid}  查询进度
- GET  /api/alpha/alpha-gpt/results/{sid}  获取结果
- POST /api/alpha/alpha-gpt/stop/{sid}  停止
- GET  /api/alpha/alpha-gpt/list  历史会话列表

WebSocket:
- WS /api/alpha/alpha-gpt/stream/{sid}  实时事件流
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..services.alpha_gpt_service import alpha_gpt_service

router = APIRouter()
logger = logging.getLogger(__name__)


class AlphaGptGenerateRequest(BaseModel):
    objective: str = Field(..., description="研究目标")
    data_path: Optional[str] = Field(None, description="数据路径（None=合成数据）")
    iterations: int = Field(5, ge=1, le=20, description="迭代轮次")
    pool_size: int = Field(10, ge=1, le=50, description="每轮想法/公式数量")
    top_k: int = Field(10, ge=1, le=50, description="最终 top-K")
    forward_returns: List[int] = Field([1, 5, 20], description="前瞻期列表")
    llm_provider: str = Field("mock", description="LLM provider")
    enable_backtest: bool = Field(False, description="启用 Trading 回测")


@router.post("/generate")
async def generate(req: AlphaGptGenerateRequest) -> Dict[str, Any]:
    """启动 Alpha-GPT 工作流（async）"""
    session_id = await alpha_gpt_service.create_session(
        objective=req.objective,
        data_path=req.data_path,
        iterations=req.iterations,
        pool_size=req.pool_size,
        top_k=req.top_k,
        forward_returns=req.forward_returns,
        llm_provider=req.llm_provider,
        enable_backtest=req.enable_backtest,
    )
    return {"session_id": session_id, "status": "pending"}


@router.get("/status/{session_id}")
async def status(session_id: str) -> Dict[str, Any]:
    """查询会话进度"""
    s = alpha_gpt_service.get_status(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    return s


@router.get("/results/{session_id}")
async def results(session_id: str) -> Dict[str, Any]:
    """获取会话结果"""
    r = alpha_gpt_service.get_results(session_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    return r


@router.post("/stop/{session_id}")
async def stop(session_id: str) -> Dict[str, Any]:
    """停止会话"""
    ok = alpha_gpt_service.stop_session(session_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"cannot stop session {session_id} (not running or not found)",
        )
    return {"session_id": session_id, "status": "stopped"}


@router.get("/list")
async def list_sessions() -> Dict[str, Any]:
    """历史会话列表"""
    return {"sessions": alpha_gpt_service.list_sessions()}


# ==============================================================================
# WebSocket 流式端点
# ==============================================================================


@router.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    """实时事件流（WebSocket）

    协议：
    - Client opens: ws://host/api/alpha/alpha-gpt/stream/{sid}
    - Server 立即发送 buffered events（包含历史 events）
    - Server 持续推送新 events 直到 session 完成
    - Server 发送 {"type": "done"} 后关闭连接

    Events:
    - round_started: 一轮开始（含 total_rounds）
    - round_completed: 一轮结束（含 formulas_evaluated, best_ir）
    - final_pool_ready: 最终 top-K 就绪
    - error: 出错（含 message）
    - done: 工作流结束
    """
    await websocket.accept()
    session = alpha_gpt_service.get_session(session_id)
    if session is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"session not found: {session_id}",
        }))
        await websocket.close()
        return

    queue = alpha_gpt_service.subscribe(session_id)
    if queue is None:
        await websocket.close()
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # 30s 无事件：发心跳
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
        alpha_gpt_service.unsubscribe(session_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
