# coding=utf-8
"""
会话管理

Session持久化与历史管理
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List
import json


@dataclass
class Session:
    """会话数据"""

    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """添加消息到历史"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        msg.update(kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """从字典反序列化"""
        data = dict(data)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class SessionManager:
    """会话管理器"""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace) / "sessions"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        """获取会话，不存在则创建"""
        if session_id in self._cache:
            return self._cache[session_id]

        session_file = self.workspace / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, "r", encoding="utf-8") as f:
                session = Session.from_dict(json.load(f))
                self._cache[session_id] = session
                return session

        session = Session(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._cache[session_id] = session
        return session

    def save_session(self, session: Session) -> None:
        """保存会话到文件"""
        session_file = self.workspace / f"{session.session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        self._cache[session.session_id] = session

    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return sorted([f.stem for f in self.workspace.glob("*.json")])

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        session_file = self.workspace / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            if session_id in self._cache:
                del self._cache[session_id]
            return True
        return False
