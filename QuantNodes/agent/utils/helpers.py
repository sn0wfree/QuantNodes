# coding=utf-8
"""
工具函数

文本处理、Token计数等
"""

import asyncio
from typing import Any, Callable, Coroutine


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """截断文本到指定长度"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def count_tokens(text: str) -> int:
    """估算token数量（简化版）"""
    return len(text) // 4


async def ensure_async(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """确保函数异步执行"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
