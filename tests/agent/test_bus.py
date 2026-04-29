# coding=utf-8
"""
测试消息总线
"""

import asyncio
from datetime import datetime
from QuantNodes.agent.bus import InboundMessage, OutboundMessage, MessageBus


class TestInboundMessage:
    def test_session_key(self):
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="chat1",
            content="hello",
        )
        assert msg.session_key == "cli:chat1"

    def test_session_key_override(self):
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="chat1",
            content="hello",
            session_key_override="custom",
        )
        assert msg.session_key == "custom"


class TestMessageBus:
    def test_inbound_queue(self):
        async def _test():
            bus = MessageBus()
            msg = InboundMessage(
                channel="cli",
                sender_id="user",
                chat_id="chat1",
                content="hello",
            )
            await bus.publish_inbound(msg)
            assert bus.inbound_size == 1
            received = await bus.consume_inbound()
            assert received.content == "hello"
            assert bus.inbound_size == 0

        asyncio.run(_test())

    def test_outbound_queue(self):
        async def _test():
            bus = MessageBus()
            msg = OutboundMessage(
                channel="cli",
                chat_id="chat1",
                content="response",
            )
            await bus.publish_outbound(msg)
            assert bus.outbound_size == 1
            received = await bus.consume_outbound()
            assert received.content == "response"

        asyncio.run(_test())
