# coding=utf-8
"""Tests for v3.0.0 Stage 5.3 NanobotRuntime (single-process lifespan).

The runtime is the bridge between FastAPI's lifespan and nanobot's
asyncio primitives. We test:

- ``init_runtime()`` returns a singleton (idempotent)
- ``start()`` gracefully degrades to ``state=unavailable`` when nanobot
  is not installed
- ``stop()`` is idempotent (can be called when never started)
- ``status()`` returns a serializable dict matching the documented
  contract (state, available, hint, error, components, ...)
- The HTTP endpoints (``/api/agent/status`` etc.) return the right
  status codes and body shape for the unavailable case

For the "with nanobot" path we use ``unittest.mock.patch`` on
``api.services.nanobot_runtime._build_components`` directly. This avoids
sys.modules gymnastics and is reliable across test ordering.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_runtime_singleton():
    """Reset the global runtime singleton before and after each test."""
    from api.services import nanobot_runtime
    nanobot_runtime._runtime = None
    yield
    nanobot_runtime._runtime = None


# ----------------------------------------------------------------------------
# Singleton lifecycle
# ----------------------------------------------------------------------------

def test_init_runtime_returns_singleton():
    from api.services.nanobot_runtime import init_runtime, get_runtime
    rt1 = init_runtime(workspace=tempfile.mkdtemp())
    rt2 = init_runtime(workspace=tempfile.mkdtemp())
    assert rt1 is rt2
    assert get_runtime() is rt1


def test_init_runtime_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOBOT_GATEWAY_PORT", "19999")
    monkeypatch.setenv("NANOBOT_GATEWAY_HOST", "10.0.0.1")
    monkeypatch.setenv("NANOBOT_WORKSPACE", str(tmp_path))
    from api.services.nanobot_runtime import init_runtime
    rt = init_runtime()
    assert rt.gateway_port == 19999
    assert rt.gateway_host == "10.0.0.1"
    assert Path(rt.workspace) == tmp_path


def test_shutdown_runtime_clears_singleton():
    from api.services.nanobot_runtime import init_runtime, get_runtime, shutdown_runtime
    init_runtime(workspace=tempfile.mkdtemp())
    assert get_runtime() is not None
    asyncio.run(shutdown_runtime())
    assert get_runtime() is None


# ----------------------------------------------------------------------------
# start() — no nanobot installed
# ----------------------------------------------------------------------------

def test_start_when_nanobot_unavailable():
    """When nanobot-ai is not installed, start() returns immediately with state=unavailable."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed — covered by other tests"):
        return

    from api.services.nanobot_runtime import init_runtime
    rt = init_runtime(workspace=tempfile.mkdtemp())
    asyncio.run(rt.start())

    s = rt.status()
    assert s["available"] is False
    assert s["state"] == "unavailable"
    assert "pip install" in (s["hint"] or "")
    assert s["components"] == {}


def test_stop_when_never_started_is_noop():
    """stop() on a runtime that was never started should not raise."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return
    from api.services.nanobot_runtime import init_runtime
    rt = init_runtime(workspace=tempfile.mkdtemp())
    asyncio.run(rt.stop())
    s = rt.status()
    assert s["state"] == "stopped"


# ----------------------------------------------------------------------------
# start() — with mocked nanobot
# ----------------------------------------------------------------------------

def test_start_with_mocked_nanobot():
    """When nanobot is available, start() builds components + schedules tasks."""
    from QuantNodes import agent as qa_mod

    # Force NANOBOT_AVAILABLE=True for this test
    original = qa_mod.NANOBOT_AVAILABLE
    qa_mod.NANOBOT_AVAILABLE = True
    try:
        from api.services.nanobot_runtime import NanobotRuntime

        # Build mocks for the components the runtime wires together.
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_agent.stop = MagicMock()
        mock_cron = MagicMock()
        mock_cron.start = AsyncMock()
        mock_cron.stop = MagicMock()
        mock_channels = MagicMock()
        mock_channels.start_all = AsyncMock()
        mock_channels.stop_all = AsyncMock()
        mock_provider_snapshot = MagicMock()
        mock_provider_snapshot.provider = MagicMock()
        mock_provider_snapshot.model = "gpt-4o"
        mock_provider_snapshot.context_window_tokens = 128000

        async def _fake_build_components(self):
            self._bus = MagicMock()
            self._session_manager = MagicMock()
            self._cron = mock_cron
            self._agent = mock_agent
            self._channels = mock_channels

        with patch.object(
            NanobotRuntime, "_build_components", _fake_build_components
        ):
            rt = NanobotRuntime(workspace=tempfile.mkdtemp(), gateway_port=18080)
            asyncio.run(rt.start())

            s = rt.status()
            assert s["state"] == "running", f"expected running, got {s}"
            assert s["available"] is True
            assert s["gateway_port"] == 18080
            assert s["components"]["agent"] is True
            assert s["components"]["cron"] is True
            assert s["components"]["channels"] is True

            # All 3 background tasks should have been created
            assert len(rt._tasks) == 3
            task_names = sorted(t.get_name() for t in rt._tasks)
            assert task_names == ["nanobot-agent", "nanobot-channels", "nanobot-cron"]
    finally:
        qa_mod.NANOBOT_AVAILABLE = original


def test_start_handles_provider_error_gracefully():
    """If _build_components raises, start() should transition to state=error with a clear hint."""
    from QuantNodes import agent as qa_mod
    original = qa_mod.NANOBOT_AVAILABLE
    qa_mod.NANOBOT_AVAILABLE = True
    try:
        from api.services.nanobot_runtime import NanobotRuntime

        async def _fake_build_components_with_error(self):
            raise RuntimeError("missing API key")

        with patch.object(
            NanobotRuntime, "_build_components", _fake_build_components_with_error
        ):
            rt = NanobotRuntime(workspace=tempfile.mkdtemp())
            asyncio.run(rt.start())
            s = rt.status()
            assert s["state"] == "error"
            assert "missing API key" in (s["error"] or "")
            assert "QUANTNODES__LLM__" in (s["hint"] or "")
    finally:
        qa_mod.NANOBOT_AVAILABLE = original


def test_stop_cancels_tasks_in_order():
    """stop() should call cron.stop, agent.stop, channels.stop_all, agent.close_mcp in order."""
    from QuantNodes import agent as qa_mod
    original = qa_mod.NANOBOT_AVAILABLE
    qa_mod.NANOBOT_AVAILABLE = True
    try:
        from api.services.nanobot_runtime import NanobotRuntime

        call_order: list[str] = []
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        mock_agent.stop = MagicMock(side_effect=lambda: call_order.append("agent.stop"))
        mock_agent.close_mcp = AsyncMock(side_effect=lambda: call_order.append("agent.close_mcp"))
        mock_cron = MagicMock()
        mock_cron.start = AsyncMock()
        mock_cron.stop = MagicMock(side_effect=lambda: call_order.append("cron.stop"))
        mock_channels = MagicMock()
        mock_channels.start_all = AsyncMock()
        mock_channels.stop_all = AsyncMock(side_effect=lambda: call_order.append("channels.stop_all"))

        async def _fake_build_components(self):
            self._bus = MagicMock()
            self._session_manager = MagicMock()
            self._cron = mock_cron
            self._agent = mock_agent
            self._channels = mock_channels

        with patch.object(
            NanobotRuntime, "_build_components", _fake_build_components
        ):
            rt = NanobotRuntime(workspace=tempfile.mkdtemp())
            asyncio.run(rt.start())
            # Wipe the background tasks bound to the start() event loop
            # so stop() doesn't try to cancel them on a closed loop.
            # The order we care about is on the synchronous components
            # (cron / agent / channels), not the asyncio.Task wrappers.
            rt._tasks = []
            asyncio.run(rt.stop())
            assert call_order == [
                "cron.stop", "agent.stop", "channels.stop_all", "agent.close_mcp",
            ]
            assert rt.status()["state"] == "stopped"
    finally:
        qa_mod.NANOBOT_AVAILABLE = original


# ----------------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------------

def test_http_status_endpoint_when_nanobot_unavailable():
    """GET /api/agent/status returns 200 with available=false hint when nanobot missing."""
    if pytest.importorskip("nanobot", reason="nanobot-ai installed"):
        return

    from contextlib import asynccontextmanager

    from api.routers import agent as agent_router
    from api.services.nanobot_runtime import init_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = init_runtime(workspace=tempfile.mkdtemp())
        await rt.start()
        app.state.nanobot_runtime = rt
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(agent_router.router, prefix="/api/agent")

    with TestClient(app) as c:
        r = c.get("/api/agent/status")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert body["state"] == "unavailable"

        r = c.get("/api/agent/health")
        assert r.status_code == 503
        body = r.json()
        assert body["available"] is False

        r = c.post("/api/agent/chat/send", json={"message": "hi"})
        assert r.status_code == 503
        assert "pip install" in str(r.json())

        r = c.post("/api/agent/restart")
        assert r.status_code == 503

        r = c.get("/api/agent/sessions")
        assert r.status_code == 503
