from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..schemas.agent import ChatMessage, ChatResponse
from ..services.agent_service import agent_service

router = APIRouter()


@router.get("/agent/status")
async def get_status():
    """Get Agent status"""
    return {
        "status": "ready" if agent_service._agent is not None else "initializing",
        "initialized": agent_service._agent is not None,
    }


@router.post("/chat", response_model=ChatResponse)
async def send_message(message: ChatMessage):
    result = await agent_service.send_message(
        content=message.content,
        session_id=message.session_id or "default",
    )
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
            
            # Stream response
            async for chunk in agent_service.stream_message(
                content=content,
                session_id=session_id,
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
