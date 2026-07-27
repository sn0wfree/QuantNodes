# coding=utf-8
"""v1 策略版本管理 (原始CICC复现, Stage 8)."""
from __future__ import annotations

from .portfolio_v1 import RotationConfig_v1


def v1_0_0() -> RotationConfig_v1:
    """v1.0.0: Stage 8 baseline (原始CICC复现).

    4 步组合管理: 去重 + 剔高相关, 强制分散, 逆波动加权, 止损 + 补位.
    Calmar ~0.78, DD -21%, Ann 16%.

    用途: 学术研究, 理想化基准, 与 CICC 报告对比.
    """
    return RotationConfig_v1(lookback=90, top_n=10)


VERSIONS_v1: dict[str, callable] = {
    "1.0.0": v1_0_0,
}

LATEST_v1 = "1.0.0"


def get_version_v1(version: str = LATEST_v1) -> RotationConfig_v1:
    """获取 v1 指定版本的配置."""
    if version not in VERSIONS_v1:
        available = ", ".join(sorted(VERSIONS_v1.keys(), reverse=True))
        raise ValueError(f"未知 v1 版本: {version!r}. 可用: {available}")
    return VERSIONS_v1[version]()


__all__ = [
    "v1_0_0",
    "VERSIONS_v1",
    "LATEST_v1",
    "get_version_v1",
]
