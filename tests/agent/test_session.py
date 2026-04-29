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

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.get_session("to_delete")
            manager.save_session(session)

            assert manager.delete_session("to_delete") is True
            assert not (manager.workspace / "to_delete.json").exists()
            assert "to_delete" not in manager._cache

    def test_delete_nonexistent_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            assert manager.delete_session("nonexistent") is False

    def test_corrupted_session_file(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session_file = manager.workspace / "corrupted.json"

            with open(session_file, "w") as f:
                f.write("this is not valid json {{{")

            try:
                manager.get_session("corrupted")
            except json.JSONDecodeError:
                pass

    def test_empty_session_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session_file = manager.workspace / "empty.json"
            session_file.touch()

            try:
                manager.get_session("empty")
            except Exception:
                pass

    def test_session_to_dict_from_dict(self):
        from datetime import datetime
        session = Session(
            session_id="test",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        session.add_message("user", "hello")

        data = session.to_dict()
        assert data["session_id"] == "test"
        assert "created_at" in data
        assert "updated_at" in data
        assert len(data["messages"]) == 1

        session2 = Session.from_dict(data)
        assert session2.session_id == "test"
        assert len(session2.messages) == 1

    def test_session_metadata(self):
        from datetime import datetime
        session = Session(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"model": "gpt-4", "tokens": 1000},
        )
        assert session.metadata["model"] == "gpt-4"

        data = session.to_dict()
        session2 = Session.from_dict(data)
        assert session2.metadata["model"] == "gpt-4"

    def test_add_message_with_extra_fields(self):
        from datetime import datetime
        session = Session(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        session.add_message("assistant", "hello", tool_calls=[{"id": "tc1", "name": "echo"}])
        assert "tool_calls" in session.messages[0]
        assert session.messages[0]["tool_calls"][0]["name"] == "echo"

    def test_session_cache_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session1 = manager.get_session("cache_test")
            session1.add_message("user", "message1")
            manager.save_session(session1)

            session2 = manager.get_session("cache_test")
            assert session2 is session1
            assert len(session2.messages) == 1

    def test_large_session_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.get_session("large")

            for i in range(100):
                session.add_message("user", f"message {i}")
                session.add_message("assistant", f"response {i}")

            manager.save_session(session)

            manager2 = SessionManager(Path(tmpdir))
            reloaded = manager2.get_session("large")
            assert len(reloaded.messages) == 200
