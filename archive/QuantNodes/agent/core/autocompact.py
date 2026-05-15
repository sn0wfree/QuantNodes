# coding=utf-8
"""
自动压缩模块（简化版）

Phase 1: 仅提供基础的上下文裁剪功能
Phase F: truncate_history 返回 (保留, 丢弃) 元组
"""

from typing import List, Dict, Any, Tuple


def truncate_history(
    messages: List[Dict[str, Any]],
    max_messages: int = 20,
    keep_system: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """裁剪历史消息，返回 (保留的消息, 被丢弃的消息)"""
    if len(messages) <= max_messages:
        return messages, []

    system_msgs = []
    other_msgs = []

    for msg in messages:
        if keep_system and msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            other_msgs.append(msg)

    dropped = []
    if len(other_msgs) > max_messages:
        dropped = other_msgs[:-max_messages]
        other_msgs = other_msgs[-max_messages:]

    return system_msgs + other_msgs, dropped


def microcompact(
    messages: List[Dict[str, Any]],
    max_tool_result_chars: int = 500,
) -> List[Dict[str, Any]]:
    """微压缩 - 缩短工具结果长度"""
    from ..utils.helpers import truncate_text

    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_tool_result_chars:
                msg = dict(msg)
                msg["content"] = truncate_text(content, max_tool_result_chars)
        result.append(msg)

    return result
