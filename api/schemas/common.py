from pydantic import BaseModel
from typing import Optional, List


class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    message_id: str
    content: str
    tools_used: List[str] = []
    usage: dict = {}


class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: Optional[dict] = None


class StatsResponse(BaseModel):
    factors: int = 0
    strategies: int = 0
    backtests: int = 0
    insights: int = 0


class ActivityItem(BaseModel):
    type: str
    name: str
    category: str = ""
    updated_at: str = ""
