# coding=utf-8
"""
异步消息队列

解耦渠道与Agent核心
"""

import asyncio

from .events import InboundMessage, OutboundMessage


class MessageBus:
    """异步消息总线"""

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """发布消息到Agent"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """消费下一条入站消息（阻塞直到可用）"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """发布响应到渠道"""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """消费下一条出站消息（阻塞直到可用）"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """待处理入站消息数量"""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """待处理出站消息数量"""
        return self.outbound.qsize()
