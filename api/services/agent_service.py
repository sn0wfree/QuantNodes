"""
Agent Service - Bridge between FastAPI and QuantNodes Agent system
"""

import asyncio
import uuid
from typing import AsyncGenerator, Optional

from QuantNodes.agent import Agent

MAX_SESSION_MESSAGES = 100


class AgentService:
    """Agent service for API layer"""

    def __init__(self, workspace: str = ".quant_agent"):
        self.workspace = workspace
        self._agent: Optional[Agent] = None

    def _get_agent(self, config: dict = None) -> Agent:
        """Get or create Agent instance"""
        if self._agent is None:
            if config is None:
                config = self._load_settings_config()
            self._agent = Agent(workspace=self.workspace, config=config)
        return self._agent

    @property
    def session_manager(self):
        """获取 Agent 内部的 SessionManager"""
        agent = self._get_agent()
        return agent.loop.session_manager

    def _load_settings_config(self) -> dict:
        """Load agent config from settings service"""
        try:
            from .settings_service import settings_service
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    settings = pool.submit(
                        asyncio.run, settings_service.get_settings()
                    ).result()
            else:
                settings = loop.run_until_complete(settings_service.get_settings())
            return settings.get("agent", {})
        except Exception:
            return {}

    def reload_agent(self) -> None:
        """Destroy cached agent and recreate with current settings"""
        self._agent = None
        self._get_agent()
        import logging
        logging.getLogger(__name__).info("Agent reloaded with current settings")

    async def send_message(
        self,
        content: str,
        session_id: str = "default",
        config: dict = None,
    ) -> dict:
        """Send message and get response (non-streaming)"""
        agent = self._get_agent(config)

        try:
            response = await agent.run(content, session_id=session_id)

            return {
                "message_id": f"msg-{uuid.uuid4().hex[:12]}",
                "content": response,
                "tools_used": [],
                "usage": {},
            }
        except Exception as e:
            return {
                "message_id": "msg-error",
                "content": f"Error: {str(e)}",
                "tools_used": [],
                "usage": {},
                "error": str(e),
            }

    async def stream_message(
        self,
        content: str,
        session_id: str = "default",
        config: dict = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream message chunks via WebSocket"""
        agent = self._get_agent(config)
        model = config.get("model") if config else None
        max_tokens = config.get("max_tokens") if config else None
        message_id = f"msg-{uuid.uuid4().hex[:12]}"

        try:
            full_content = ""
            tools_used = []
            async for event in agent.chat(content, session_id=session_id, model=model, max_tokens=max_tokens):
                event["message_id"] = message_id

                if event["type"] == "token":
                    full_content += event.get("content", "")
                    yield event
                elif event["type"] == "tool_call":
                    tools_used.append(event.get("name", ""))
                    yield event
                elif event["type"] == "tool_result":
                    yield event
                elif event["type"] == "done":
                    final = event.get("content", "")
                    if final:
                        full_content = final
                    yield {
                        "type": "done",
                        "message_id": message_id,
                        "content": full_content,
                        "tools_used": list(set(tools_used)),
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    }
                elif event["type"] == "error":
                    yield event

        except Exception as e:
            yield {
                "type": "error",
                "content": str(e),
                "message_id": message_id,
            }

    def get_history(self, session_id: str) -> list:
        """Get chat history for session"""
        session = self.session_manager.get_session(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in session.messages]

    def clear_history(self, session_id: str) -> None:
        """Clear chat history for session"""
        self.session_manager.delete_session(session_id)

    def list_sessions(self) -> list[dict]:
        """List all sessions with metadata"""
        sessions = []
        for info in self.session_manager.list_sessions_with_info():
            sid = info["session_id"]
            session = self.session_manager.get_session(sid)
            first_msg = session.messages[0] if session.messages else None
            last_msg = session.messages[-1] if session.messages else None
            sessions.append({
                "session_id": sid,
                "message_count": info["message_count"],
                "created_at": info.get("created_at", ""),
                "updated_at": info.get("updated_at", ""),
                "first_message": first_msg.get("content", "")[:100] if first_msg else "",
                "last_message": last_msg.get("content", "")[:100] if last_msg else "",
            })
        return sessions

    def create_session(self, session_id: str | None = None) -> dict:
        """Create a new session"""
        if session_id is None:
            session_id = f"session-{uuid.uuid4().hex[:8]}"
        session = self.session_manager.get_session(session_id)
        self.session_manager.save_session(session)
        return {"session_id": session_id, "message_count": 0}

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.session_manager.delete_session(session_id)


# Singleton instance
agent_service = AgentService()
