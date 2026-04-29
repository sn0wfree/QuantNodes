# coding=utf-8
"""
测试会话管理
"""

import tempfile
from pathlib import Path
from QuantNodes.agent.session import Session, SessionManager


class TestSession:
    def test_add_message(self):
        session = Session(
            session_id="test",
            created_at="2024-01-01",
            updated_at="2024-01-01",
        )
        session.add_message("user", "hello")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "hello"


class TestSessionManager:
    def test_get_create_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.get_session("new_session")
            assert session.session_id == "new_session"
            assert len(session.messages) == 0

    def test_save_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.get_session("test")
            session.add_message("user", "hello")
            manager.save_session(session)

            manager2 = SessionManager(Path(tmpdir))
            session2 = manager2.get_session("test")
            assert len(session2.messages) == 1
            assert session2.messages[0]["content"] == "hello"

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            manager.get_session("session1")
            manager.get_session("session2")
            manager.save_session(manager.get_session("session1"))
            manager.save_session(manager.get_session("session2"))

            sessions = manager.list_sessions()
            assert len(sessions) == 2
            assert "session1" in sessions
            assert "session2" in sessions
