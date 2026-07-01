# coding=utf-8
"""
core/events.py — 轻量级进程内事件总线 (Tier 0: Foundation)

替代 v2.x 中已删除的 agent/bus/ 设计 (2026-06-23 nanobot 重构遗留)。

设计原则:
- 同步语义: handler 在 publish 线程同步执行
- 单进程: 不支持跨进程/分布式
- 解耦: publisher 与 subscriber 互不感知
- 失败隔离: handler 异常不影响 publisher 和其他 handler

使用示例:
    from QuantNodes.core.events import get_event_bus, Events

    bus = get_event_bus()

    def on_factor_mined(event):
        logger.info("New factor: %s", event.payload["formula"])

    bus.subscribe(Events.FACTOR_MINED, on_factor_mined)
    bus.publish_sync(Events.FACTOR_MINED, formula="rank(close)", ir=0.5)

Event 命名规范:
    `<domain>.<action>` — 例如 factor.mined / backtest.completed
    所有事件常量定义在 Events 类中 (避免散落的字符串字面量)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """事件对象

    Args:
        name: 事件名 (建议从 Events 类选取)
        payload: 事件数据 (字典形式)
        source: 触发源标识 (可选, 用于调试)
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Event.name must be a non-empty string")


HandlerType = Callable[[Event], None]


class EventBus:
    """进程内事件总线 (同步)

    线程安全: 本类内部无锁，订阅/发布应在同一线程或加外部锁。
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[HandlerType]] = {}

    def subscribe(self, event_name: str, handler: HandlerType) -> None:
        """订阅事件。

        同一 handler 可多次订阅同一事件, 触发时执行多次。
        """
        self._subscribers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: HandlerType) -> None:
        """取消订阅。handler 不存在时静默忽略。"""
        if event_name in self._subscribers:
            try:
                self._subscribers[event_name].remove(handler)
            except ValueError:
                pass

    def publish(self, event: Event) -> None:
        """发布事件。同步调用所有订阅者。

        异常隔离: 单个 handler 抛异常不影响其他 handler 和 publisher。
        """
        handlers = self._subscribers.get(event.name, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(
                    "Event handler %r failed for %s: %s",
                    handler,
                    event.name,
                    e,
                )

    def publish_sync(self, name: str, source: str = "", **payload: Any) -> None:
        """便捷发布接口。"""
        self.publish(Event(name=name, payload=payload, source=source))

    def clear(self) -> None:
        """清空所有订阅 (主要用于测试)。"""
        self._subscribers.clear()

    def subscriber_count(self, event_name: str) -> int:
        """返回某事件的订阅者数量 (主要用于测试)。"""
        return len(self._subscribers.get(event_name, []))

    def event_names(self) -> List[str]:
        """返回所有已注册的事件名 (主要用于测试)。"""
        return list(self._subscribers.keys())


# 全局单例
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局 EventBus 单例。

    延迟创建: 首次调用时实例化, 后续返回同一对象。
    测试中可用 reset_event_bus() 重置。
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """重置全局 EventBus (主要用于测试隔离)。"""
    global _global_bus
    _global_bus = None


class Events:
    """事件名常量集中定义

    命名规范: `<domain>.<action>`
    - factor.*    因子挖掘相关
    - backtest.*  回测相关
    - agent.*     Agent 交互相关
    - monitor.*   监控相关
    - workflow.*  工作流相关
    """

    # factor 事件
    FACTOR_MINED = "factor.mined"
    FACTOR_EVALUATED = "factor.evaluated"
    FACTOR_REJECTED = "factor.rejected"

    # backtest 事件
    BACKTEST_COMPLETED = "backtest.completed"
    BACKTEST_FAILED = "backtest.failed"

    # agent 事件
    AGENT_MESSAGE = "agent.message"
    AGENT_TOOL_CALLED = "agent.tool_called"

    # monitor 事件
    DRIFT_DETECTED = "monitor.drift_detected"

    # workflow 事件
    WORKFLOW_ROUND_START = "workflow.round_start"
    WORKFLOW_ROUND_END = "workflow.round_end"


__all__ = [
    "Event",
    "EventBus",
    "Events",
    "get_event_bus",
    "reset_event_bus",
]