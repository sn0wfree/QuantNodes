# coding=utf-8
"""
路径检查模块

提供项目边界检查功能。
"""

from pathlib import Path
from typing import Optional


def is_within_project(filepath: str, project_root: str) -> bool:
    """检查文件路径是否在项目目录内

    Args:
        filepath: 文件路径
        project_root: 项目根目录

    Returns:
        True if within project, False otherwise
    """
    try:
        file_path = Path(filepath).resolve()
        project_path = Path(project_root).resolve()
        return file_path.is_relative_to(project_path)
    except (ValueError, OSError):
        return False


def assert_within_project(filepath: str, project_root: str) -> None:
    """断言文件路径在项目目录内

    Raises:
        ExternalDirectoryError: 如果文件在项目外
    """
    if not is_within_project(filepath, project_root):
        raise ExternalDirectoryError(
            f"Access to external directory denied: {filepath}"
        )


class ExternalDirectoryError(Exception):
    """外部目录访问错误"""
    pass