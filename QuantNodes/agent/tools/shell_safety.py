# coding=utf-8
"""
Shell 安全模块

提供命令超时、执行控制等功能。
"""

import asyncio
import signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShellConfig:
    """Shell 执行配置"""
    timeout_seconds: int = 120
    max_output_bytes: int = 1024 * 1024
    max_output_lines: int = 10000
    project_root: Optional[str] = None


async def execute_with_timeout(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 120,
    env: Optional[dict] = None,
) -> tuple[int, str, str]:
    """执行 shell 命令（带超时）

    Args:
        command: 要执行的命令
        cwd: 工作目录
        timeout: 超时秒数
        env: 环境变量

    Returns:
        (exit_code, stdout, stderr)
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
        return proc.returncode or 0, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ShellTimeoutError(
            f"Command timed out after {timeout}s: {command[:100]}"
        )


class ShellTimeoutError(Exception):
    """Shell 命令超时"""
    pass