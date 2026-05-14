# coding=utf-8
"""
输出截断模块

提供输出长度控制功能。
"""

from dataclasses import dataclass


@dataclass
class TruncatedOutput:
    """截断后的输出"""
    content: str
    truncated: bool
    total_lines: int
    total_bytes: int
    kept_lines: int
    kept_bytes: int


def truncate_output(
    output: str,
    max_lines: int = 10000,
    max_bytes: int = 1024 * 1024,
) -> TruncatedOutput:
    """截断输出

    策略：
    1. 保留前 max_lines/2 行
    2. 保留后 max_lines/2 行
    3. 中间用省略标记替代
    4. 同时检查字节限制

    Args:
        output: 原始输出
        max_lines: 最大行数
        max_bytes: 最大字节数

    Returns:
        TruncatedOutput 对象
    """
    lines = output.split("\n")
    total_lines = len(lines)
    total_bytes = len(output.encode("utf-8"))

    truncated = False
    kept_lines = total_lines
    kept_bytes = total_bytes

    if total_lines > max_lines or total_bytes > max_bytes:
        truncated = True
        half = max_lines // 2
        kept_lines = min(total_lines, max_lines)

        if total_lines > max_lines:
            kept_lines = max_lines
            head = lines[:half]
            tail = lines[-(max_lines - half):]
            lines = head + ["... (truncated) ..."] + tail

        result = "\n".join(lines)

        if len(result.encode("utf-8")) > max_bytes:
            result = result[:max_bytes] + "\n... (truncated by bytes) ..."
            kept_bytes = max_bytes
    else:
        result = output

    return TruncatedOutput(
        content=result,
        truncated=truncated,
        total_lines=total_lines,
        total_bytes=total_bytes,
        kept_lines=kept_lines,
        kept_bytes=kept_bytes,
    )