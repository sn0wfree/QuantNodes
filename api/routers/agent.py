from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from ..schemas.agent import ChatMessage, ChatResponse
from ..services.agent_service import agent_service
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/agent/status")
async def get_status():
    """Get Agent status"""
    return {
        "status": "ready" if agent_service._agent is not None else "initializing",
        "initialized": agent_service._agent is not None,
    }


@router.post("/chat", response_model=ChatResponse)
async def send_message(message: ChatMessage):
    logger.info(f"[chat] Received request: content={message.content[:50]!r}..., session_id={message.session_id}")
    result = await agent_service.send_message(
        content=message.content,
        session_id=message.session_id or "default",
    )
    logger.info(f"[chat] Response type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
    return ChatResponse(**result)


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    return agent_service.get_history(session_id)


@router.delete("/chat/history/{session_id}")
async def clear_history(session_id: str):
    agent_service.clear_history(session_id)
    return {"status": "cleared"}


@router.get("/chat/sessions")
async def list_sessions():
    return agent_service.list_sessions()


@router.post("/chat/sessions")
async def create_session(data: dict = None):
    session_id = (data or {}).get("session_id")
    return agent_service.create_session(session_id)


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = agent_service.delete_session(session_id)
    if not deleted:
        return {"status": "not_found"}
    return {"status": "deleted"}


@router.get("/chat/export/{session_id}")
async def export_session(session_id: str, format: str = "markdown"):
    """Export session as Markdown or JSON"""
    history = agent_service.get_history(session_id)

    if format == "json":
        import json
        return PlainTextResponse(
            json.dumps(history, indent=2, ensure_ascii=False),
            media_type="application/json",
        )

    # Markdown format
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# QuantNodes Chat Export",
        f"Session: {session_id} | Date: {now}",
        "",
        "---",
        "",
    ]
    for msg in history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")

    return PlainTextResponse("\n".join(lines), media_type="text/markdown")


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = "default"
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Validate input
            if not isinstance(data, dict):
                await websocket.send_json({"type": "error", "content": "Invalid message format"})
                continue
            
            content = data.get("content", "")
            if not isinstance(content, str) or not content.strip():
                await websocket.send_json({"type": "error", "content": "Content must be a non-empty string"})
                continue
            
            session_id = data.get("session_id", session_id)
            model = data.get("model")
            max_tokens = data.get("max_tokens")
            mode = data.get("mode")
            
            # Stream response
            async for chunk in agent_service.stream_message(
                content=content,
                session_id=session_id,
                config={"model": model, "max_tokens": max_tokens, "mode": mode} if model or max_tokens or mode else None,
            ):
                await websocket.send_json(chunk)
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
