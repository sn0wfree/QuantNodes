# coding=utf-8
"""Agent HTTP endpoints — status / chat / restart.

v3.0.0 Stage 5.3: All endpoints gracefully degrade when
``nanobot-ai`` is not installed. They return a clear ``{available: false,
hint: "..."}`` body and a 503 status, so the frontend can show an
install prompt instead of crashing.

Routes:
- ``GET  /api/agent/status``      — runtime state (always 200)
- ``GET  /api/agent/health``      — readiness probe (200/503)
- ``POST /api/agent/restart``     — destroy and rebuild the runtime
- ``POST /api/agent/chat/send``   — non-streaming chat (delegates to nanobot AgentLoop)
- ``GET  /api/agent/sessions``    — list websocket session keys
- ``DELETE /api/agent/sessions/{key}`` — delete a websocket session
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from QuantNodes.agent import NANOBOT_AVAILABLE, NanobotNotInstalled

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: str = Field("default", description="WebSocket session key (e.g. 'unified:default')")
    model: Optional[str] = Field(None, description="Override the model for this turn")


class ChatResponse(BaseModel):
    message_id: str
    content: str
    tools_used: List[str] = Field(default_factory=list)
    session_id: str
    stop_reason: str = "stop"
    error: Optional[str] = None


class RestartResponse(BaseModel):
    success: bool
    state: str
    message: str


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _runtime(request: Request):
    """Return the NanobotRuntime from app.state, or raise 503."""
    rt = getattr(request.app.state, "nanobot_runtime", None)
    if rt is None:
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "hint": "nanobot runtime not initialized (server starting up)",
            },
        )
    return rt


def _unavailable_response(hint: Optional[str] = None) -> JSONResponse:
    """Return a standard 503 'nanobot unavailable' response."""
    body = {
        "available": False,
        "state": "unavailable",
        "hint": hint or (
            "nanobot-ai is not installed. Run:  "
            "pip install 'quantnodes[agent]'  (or  'quantnodes[all]')"
        ),
    }
    return JSONResponse(status_code=503, content=body)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@router.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    """Return the current nanobot runtime state.

    Always returns 200 (even when the runtime is unavailable) so the
    frontend can display the install hint without treating it as an
    HTTP error.
    """
    rt = getattr(request.app.state, "nanobot_runtime", None)
    if rt is None:
        return {
            "available": False,
            "state": "uninitialized",
            "hint": "nanobot runtime not initialized",
            "components": {},
        }
    return rt.status()


@router.get("/health")
async def get_health(request: Request):
    """Readiness probe — 200 when runtime is running, 503 otherwise."""
    rt = getattr(request.app.state, "nanobot_runtime", None)
    if rt is None or rt.state.state != "running":
        return _unavailable_response(
            hint=f"nanobot runtime state={rt.state.state if rt else 'None'}"
        )
    return {"status": "ok", "state": "running"}


@router.post("/restart", response_model=RestartResponse)
async def restart_runtime(request: Request) -> RestartResponse:
    """Destroy the current runtime and start a fresh one.

    Useful after editing .env (LLM_API_KEY, etc.) — the new env is read
    on startup.
    """
    if not NANOBOT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "hint": "nanobot-ai not installed. pip install 'quantnodes[agent]'",
            },
        )

    rt = _runtime(request)
    try:
        await rt.stop()
        await rt.start()
        return RestartResponse(
            success=True,
            state=rt.state.state,
            message=f"nanobot runtime restarted (state={rt.state.state})",
        )
    except Exception as e:
        logger.exception("Failed to restart nanobot runtime")
        raise HTTPException(
            status_code=500,
            detail=f"restart failed: {type(e).__name__}: {e}",
        )


@router.post("/chat/send", response_model=ChatResponse)
async def send_chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Send a chat message and get the final response (non-streaming).

    For streaming, use the WebSocket channel directly
    (``ws://localhost:{gateway_port}/``). This endpoint is a convenience
    for HTTP-only clients (curl, scripts).
    """
    if not NANOBOT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "hint": "nanobot-ai not installed. pip install 'quantnodes[agent]'",
            },
        )

    rt = _runtime(request)
    if rt.state.state != "running" or rt.bus is None:
        raise HTTPException(
            status_code=503,
            detail={
                "available": False,
                "state": rt.state.state,
                "hint": f"nanobot runtime is {rt.state.state}, not running",
            },
        )

    try:
        from nanobot.bus.events import InboundMessage

        msg_id = f"msg-{uuid.uuid4().hex[:12]}"
        inbound = InboundMessage(
            channel="api",
            chat_id=req.session_id,
            sender_id="api",
            content=req.message,
            session_key=req.session_id,
            metadata={"_wants_stream": False, "api_message_id": msg_id},
        )
        await rt.bus.publish_inbound(inbound)
        # process_direct is the synchronous-mode API; we use it to read out
        # the final response without depending on the bus consumer's
        # coroutine ordering.
        if rt.agent is None:
            raise HTTPException(status_code=503, detail={"available": False, "hint": "agent not initialized"})
        response = await rt.agent.process_direct(
            req.message,
            session_key=req.session_id,
            channel="api",
            chat_id=req.session_id,
        )
        return ChatResponse(
            message_id=msg_id,
            content=getattr(response, "content", "") or "",
            tools_used=list(getattr(response, "tools_used", []) or []),
            session_id=req.session_id,
            stop_reason=getattr(response, "stop_reason", "stop") or "stop",
        )
    except NanobotNotInstalled as e:
        raise HTTPException(status_code=503, detail={"available": False, "hint": str(e)})
    except Exception as e:
        logger.exception("Chat failed")
        return ChatResponse(
            message_id="msg-error",
            content="",
            session_id=req.session_id,
            error=f"{type(e).__name__}: {e}",
        )


@router.get("/sessions")
async def list_sessions(request: Request) -> Dict[str, Any]:
    """List nanobot session keys (read-only metadata)."""
    if not NANOBOT_AVAILABLE:
        return _unavailable_response()

    rt = _runtime(request)
    if rt.session_manager is None:
        return {"available": True, "state": rt.state.state, "sessions": []}
    try:
        sessions = rt.session_manager.list_sessions() or []
        # Return a compact view: key, message_count, updated_at
        compact = []
        for s in sessions[:100]:
            if isinstance(s, dict):
                compact.append({
                    "key": s.get("key", ""),
                    "message_count": s.get("message_count", 0),
                    "updated_at": s.get("updated_at", ""),
                })
        return {"available": True, "state": rt.state.state, "sessions": compact}
    except Exception as e:
        logger.exception("list_sessions failed")
        return {"available": True, "state": rt.state.state, "sessions": [], "error": str(e)}


@router.delete("/sessions/{key:path}")
async def delete_session(key: str, request: Request) -> Dict[str, Any]:
    """Delete a session by key (e.g. ``websocket:default``)."""
    if not NANOBOT_AVAILABLE:
        return _unavailable_response()

    rt = _runtime(request)
    if rt.session_manager is None:
        return {"available": True, "deleted": False, "reason": "session_manager not initialized"}
    try:
        deleted = rt.session_manager.delete_session(key)
        return {"available": True, "deleted": bool(deleted), "key": key}
    except Exception as e:
        logger.exception("delete_session(%s) failed", key)
        raise HTTPException(status_code=500, detail=str(e))
