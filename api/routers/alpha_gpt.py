# coding=utf-8
"""
alpha_gpt router - Alpha-GPT 5 端点 REST API

- POST /api/alpha/alpha-gpt/generate  启动工作流
- GET  /api/alpha/alpha-gpt/status/{sid}  查询进度
- GET  /api/alpha/alpha-gpt/results/{sid}  获取结果
- POST /api/alpha/alpha-gpt/stop/{sid}  停止
- GET  /api/alpha/alpha-gpt/list  历史会话列表
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.alpha_gpt_service import alpha_gpt_service

router = APIRouter()


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
