# coding=utf-8
"""
test_alpha_gpt_websocket.py - WebSocket 流式端点测试

覆盖：
- subscribe / unsubscribe
- 事件 replay（subscribe 时立即收到 buffered events）
- REST + WebSocket 集成（启动 → 订阅 → 收事件）
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import polars as pl
import pytest

from api.services.alpha_gpt_service import (
    AlphaGptService,
    ALPHA_GPT_EVENT_TYPES,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date, "code": code, "close": close,
                "open": close, "high": close + 1, "low": close - 1,
                "vol": 1000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


class TestEventTypes:
    def test_event_types_defined(self):
        assert "round_started" in ALPHA_GPT_EVENT_TYPES
        assert "round_completed" in ALPHA_GPT_EVENT_TYPES
        assert "final_pool_ready" in ALPHA_GPT_EVENT_TYPES
        assert "done" in ALPHA_GPT_EVENT_TYPES
        assert "error" in ALPHA_GPT_EVENT_TYPES


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_unknown_session(self):
        svc = AlphaGptService()
        q = svc.subscribe("nonexistent")
        assert q is None

    @pytest.mark.asyncio
    async def test_subscribe_replays_buffered_events(self):
        svc = AlphaGptService()
        sid = await svc.create_session(
            objective="test", iterations=1, pool_size=1,
        )
        # Before subscribe: events buffer should be empty (workflow not yet run)
        s = svc.get_session(sid)
        assert s is not None

        # Emit some manual events
        await svc._emit(s, {"type": "round_started", "round": 1})
        await svc._emit(s, {"type": "round_completed", "round": 1})

        # Subscribe should replay
        q = svc.subscribe(sid)
        assert q is not None
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert len(events) == 2
        assert events[0]["type"] == "round_started"
        assert events[1]["type"] == "round_completed"

        svc.unsubscribe(sid, q)

    @pytest.mark.asyncio
    async def test_subscribe_receives_new_events(self):
        svc = AlphaGptService()
        sid = await svc.create_session(
            objective="test", iterations=1, pool_size=1,
        )
        s = svc.get_session(sid)
        q = svc.subscribe(sid)
        assert q is not None

        # Emit new event
        await svc._emit(s, {"type": "round_completed", "round": 1})

        # Drain
        event = q.get_nowait()
        assert event["type"] == "round_completed"
        assert "session_id" in event
        assert "ts" in event

        svc.unsubscribe(sid, q)


class TestEndToEndStream:
    @pytest.mark.asyncio
    async def test_full_workflow_emits_events(self, sample_data):
        """运行完整 workflow，验证事件流"""
        svc = AlphaGptService()
        sid = await svc.create_session(
            objective="test",
            iterations=2,
            pool_size=2,
            forward_returns=[1],
        )
        s = svc.get_session(sid)
        q = svc.subscribe(sid)

        # 等待 session 完成
        for _ in range(100):
            await asyncio.sleep(0.05)
            if s.status in {"completed", "failed", "stopped"}:
                break

        # Drain events
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        types_seen = {e["type"] for e in events}
        assert "round_started" in types_seen
        assert "round_completed" in types_seen
        assert "final_pool_ready" in types_seen
        assert "done" in types_seen

        svc.unsubscribe(sid, q)

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_events(self):
        svc = AlphaGptService()
        sid = await svc.create_session(
            objective="test", iterations=1, pool_size=1,
        )
        s = svc.get_session(sid)
        q = svc.subscribe(sid)
        svc.unsubscribe(sid, q)

        # Emit after unsubscribe
        await svc._emit(s, {"type": "round_started", "round": 1})

        # Should NOT receive
        assert q.empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
