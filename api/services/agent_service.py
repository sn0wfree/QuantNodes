"""
Agent Service - Bridge between FastAPI and QuantNodes Agent system
"""

import asyncio
import json
import uuid
from collections import OrderedDict
from typing import AsyncGenerator, Optional
from pathlib import Path

from QuantNodes.agent import Agent
from QuantNodes.agent.tools.wiki import WikiTool
from QuantNodes.agent.core.loop import AgentLoop

MAX_SESSION_MESSAGES = 100


class AgentService:
    """Agent service for API layer"""

    def __init__(self, workspace: str = ".quant_agent"):
        self.workspace = workspace
        self._agent: Optional[Agent] = None
        self._sessions: dict[str, list] = {}

    def _get_agent(self, config: dict = None) -> Agent:
        """Get or create Agent instance"""
        if self._agent is None:
            self._agent = Agent(workspace=self.workspace, config=config or {})
        return self._agent

    async def send_message(
        self,
        content: str,
        session_id: str = "default",
        config: dict = None,
    ) -> dict:
        """Send message and get response (non-streaming)"""
        agent = self._get_agent(config)
        
        # Store user message
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "role": "user",
            "content": content,
        })

        try:
            # Run agent
            response = await agent.run(content, session_id=session_id)
            
            # Store assistant response
            self._sessions[session_id].append({
                "role": "assistant",
                "content": response,
            })
            
            # Trim to max messages
            if len(self._sessions[session_id]) > MAX_SESSION_MESSAGES:
                self._sessions[session_id] = self._sessions[session_id][-MAX_SESSION_MESSAGES:]

            return {
                "message_id": f"msg-{uuid.uuid4().hex[:12]}",
                "content": response,
                "tools_used": [],
                "usage": {},
            }
        except Exception as e:
            return {
                "message_id": f"msg-error",
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
        message_id = f"msg-{uuid.uuid4().hex[:12]}"
        
        # Store user message
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "role": "user",
            "content": content,
        })

        try:
            # Stream agent response
            full_content = ""
            async for chunk in agent.chat(content, session_id=session_id):
                full_content += chunk
                yield {
                    "type": "chunk",
                    "content": chunk,
                    "message_id": message_id,
                }

            # Store assistant response
            self._sessions[session_id].append({
                "role": "assistant",
                "content": full_content,
            })
            
            # Trim to max messages
            if len(self._sessions[session_id]) > MAX_SESSION_MESSAGES:
                self._sessions[session_id] = self._sessions[session_id][-MAX_SESSION_MESSAGES:]

            # Send done signal
            yield {
                "type": "done",
                "message_id": message_id,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        except Exception as e:
            yield {
                "type": "error",
                "content": str(e),
                "message_id": message_id,
            }

    def get_history(self, session_id: str) -> list:
        """Get chat history for session"""
        return self._sessions.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        """Clear chat history for session"""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Singleton instance
agent_service = AgentService()
