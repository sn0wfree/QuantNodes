# coding=utf-8
"""
test_events.py - core/events.py 单元测试

覆盖:
- Event dataclass 验证
- EventBus: subscribe / unsubscribe / publish / publish_sync
- EventBus: 异常隔离 (单 handler 失败不影响其他)
- EventBus: clear / subscriber_count / event_names
- get_event_bus / reset_event_bus (全局单例)
- Events 类常量定义
"""

from __future__ import annotations

import pytest

from QuantNodes.core.events import (
    Event,
    EventBus,
    Events,
    get_event_bus,
    reset_event_bus,
)


@pytest.fixture(autouse=True)
def _reset_bus():
    """每个测试前后重置全局 EventBus，避免跨测试污染。"""
    reset_event_bus()
    yield
    reset_event_bus()


# ==============================================================================
# Event dataclass
# ==============================================================================


class TestEvent:
    def test_minimal_construction(self):
        e = Event(name="test.event")
        assert e.name == "test.event"
        assert e.payload == {}
        assert e.source == ""

    def test_full_construction(self):
        e = Event(name="factor.mined", payload={"formula": "rank(close)"}, source="pipeline")
        assert e.name == "factor.mined"
        assert e.payload == {"formula": "rank(close)"}
        assert e.source == "pipeline"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Event(name="")

    def test_non_string_name_raises(self):
        with pytest.raises(ValueError):
            Event(name=123)  # type: ignore[arg-type]


# ==============================================================================
# EventBus basic
# ==============================================================================


class TestEventBusBasic:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("foo", handler)
        bus.publish(Event(name="foo", payload={"x": 1}))

        assert len(received) == 1
        assert received[0].name == "foo"
        assert received[0].payload == {"x": 1}

    def test_publish_to_nonexistent_event_is_noop(self):
        bus = EventBus()
        # 不应抛异常
        bus.publish(Event(name="no.subscribers", payload={}))

    def test_publish_sync(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda e: received.append(e.payload))
        bus.publish_sync("test.event", a=1, b=2)
        assert received == [{"a": 1, "b": 2}]

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe("e", lambda e: a.append(e.payload))
        bus.subscribe("e", lambda e: b.append(e.payload))
        bus.publish_sync("e", x=1)
        assert a == [{"x": 1}]
        assert b == [{"x": 1}]

    def test_same_handler_subscribed_twice(self):
        bus = EventBus()
        count = []

        def handler(event):
            count.append(1)

        bus.subscribe("e", handler)
        bus.subscribe("e", handler)
        bus.publish_sync("e")
        assert len(count) == 2  # handler invoked twice


# ==============================================================================
# EventBus unsubscribe / management
# ==============================================================================


class TestEventBusUnsubscribe:
    def test_unsubscribe_existing(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("e", handler)
        bus.unsubscribe("e", handler)
        bus.publish_sync("e", x=1)
        assert received == []

    def test_unsubscribe_nonexistent_silent(self):
        bus = EventBus()
        # 不存在的 event_name — 应静默忽略
        bus.unsubscribe("no.such", lambda e: None)
        # 不存在的 handler — 应静默忽略
        bus.subscribe("e", lambda e: None)
        bus.unsubscribe("e", lambda e: None)  # 不同的 lambda

    def test_unsubscribe_one_of_many(self):
        bus = EventBus()
        a, b = [], []
        ha = lambda e: a.append(e)
        hb = lambda e: b.append(e)
        bus.subscribe("e", ha)
        bus.subscribe("e", hb)
        bus.unsubscribe("e", ha)
        bus.publish_sync("e")
        assert a == []
        assert len(b) == 1


class TestEventBusManagement:
    def test_clear(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.clear()
        assert bus.event_names() == []

    def test_subscriber_count(self):
        bus = EventBus()
        assert bus.subscriber_count("e") == 0
        bus.subscribe("e", lambda x: None)
        bus.subscribe("e", lambda x: None)
        assert bus.subscriber_count("e") == 2
        assert bus.subscriber_count("not.subscribed") == 0

    def test_event_names(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        assert set(bus.event_names()) == {"a", "b"}


# ==============================================================================
# Exception isolation
# ==============================================================================


class TestEventBusExceptionIsolation:
    def test_failing_handler_does_not_block_others(self):
        bus = EventBus()
        results = []

        def bad_handler(event):
            raise RuntimeError("boom")

        def good_handler(event):
            results.append("ok")

        bus.subscribe("e", bad_handler)
        bus.subscribe("e", good_handler)

        # bad_handler 抛异常, good_handler 应仍执行
        bus.publish_sync("e")
        assert results == ["ok"]

    def test_failing_handler_does_not_block_publisher(self):
        bus = EventBus()

        def bad_handler(event):
            raise RuntimeError("boom")

        bus.subscribe("e", bad_handler)

        # publish 本身不应抛异常
        bus.publish_sync("e", x=1)  # 不抛


# ==============================================================================
# Global singleton
# ==============================================================================


class TestGlobalBus:
    def test_get_event_bus_returns_singleton(self):
        b1 = get_event_bus()
        b2 = get_event_bus()
        assert b1 is b2

    def test_reset_event_bus_creates_new_instance(self):
        b1 = get_event_bus()
        reset_event_bus()
        b2 = get_event_bus()
        assert b1 is not b2

    def test_singleton_state_persists(self):
        bus = get_event_bus()
        received = []
        bus.subscribe("persistent", lambda e: received.append(e.payload))

        # 再次获取同一实例
        bus2 = get_event_bus()
        bus2.publish_sync("persistent", x=42)

        assert received == [{"x": 42}]


# ==============================================================================
# Events 常量
# ==============================================================================


class TestEventsConstants:
    def test_factor_events(self):
        assert Events.FACTOR_MINED == "factor.mined"
        assert Events.FACTOR_EVALUATED == "factor.evaluated"
        assert Events.FACTOR_REJECTED == "factor.rejected"

    def test_backtest_events(self):
        assert Events.BACKTEST_COMPLETED == "backtest.completed"
        assert Events.BACKTEST_FAILED == "backtest.failed"

    def test_agent_events(self):
        assert Events.AGENT_MESSAGE == "agent.message"
        assert Events.AGENT_TOOL_CALLED == "agent.tool_called"

    def test_monitor_events(self):
        assert Events.DRIFT_DETECTED == "monitor.drift_detected"

    def test_workflow_events(self):
        assert Events.WORKFLOW_ROUND_START == "workflow.round_start"
        assert Events.WORKFLOW_ROUND_END == "workflow.round_end"

    def test_all_events_unique(self):
        values = [
            Events.FACTOR_MINED,
            Events.FACTOR_EVALUATED,
            Events.FACTOR_REJECTED,
            Events.BACKTEST_COMPLETED,
            Events.BACKTEST_FAILED,
            Events.AGENT_MESSAGE,
            Events.AGENT_TOOL_CALLED,
            Events.DRIFT_DETECTED,
            Events.WORKFLOW_ROUND_START,
            Events.WORKFLOW_ROUND_END,
        ]
        assert len(values) == len(set(values))

    def test_naming_convention_domain_action(self):
        """所有事件名遵循 <domain>.<action> 规范。"""
        all_events = [
            Events.FACTOR_MINED,
            Events.FACTOR_EVALUATED,
            Events.FACTOR_REJECTED,
            Events.BACKTEST_COMPLETED,
            Events.BACKTEST_FAILED,
            Events.AGENT_MESSAGE,
            Events.AGENT_TOOL_CALLED,
            Events.DRIFT_DETECTED,
            Events.WORKFLOW_ROUND_START,
            Events.WORKFLOW_ROUND_END,
        ]
        for event_name in all_events:
            parts = event_name.split(".")
            assert len(parts) == 2, f"{event_name} 不符合 <domain>.<action> 规范"


# ==============================================================================
# Integration: Events + EventBus 协同
# ==============================================================================


class TestIntegration:
    def test_realistic_factor_mined_scenario(self):
        """模拟因子挖掘流水线中的典型事件流。"""
        bus = EventBus()
        mined_factors = []
        rejected_factors = []

        bus.subscribe(
            Events.FACTOR_MINED,
            lambda e: mined_factors.append(e.payload),
        )
        bus.subscribe(
            Events.FACTOR_REJECTED,
            lambda e: rejected_factors.append(e.payload),
        )

        # 流水线发送事件
        bus.publish_sync(
            Events.FACTOR_MINED,
            source="AlphaPipeline",
            formula="rank(close)",
            ir=0.5,
        )
        bus.publish_sync(
            Events.FACTOR_REJECTED,
            source="AlphaPipeline",
            formula="bad_formula",
            reason="compile_error",
        )

        assert len(mined_factors) == 1
        assert mined_factors[0]["formula"] == "rank(close)"
        assert mined_factors[0]["ir"] == 0.5

        assert len(rejected_factors) == 1
        assert rejected_factors[0]["reason"] == "compile_error"