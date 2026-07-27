# coding=utf-8
"""固收+ (80/20) 模块 (Stage 11 精简 stub).

提供 FixedIncomePlus 类及配套数据结构.
原本与 fi_plus.py 共存, 现在作为 stub 模块被多文件 re-export,
performance_metrics 已迁移到 common.metrics.compute_metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FixedIncomePlusConfig:
    bond_code: str = "511260"
    bond_weight: float = 0.8
    rotation: object = None
    rebalance_freq: str = "M"


@dataclass
class FixedIncomePlusResult:
    nav: pd.Series
    states: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class FixedIncomePlus:
    def __init__(self, bond_nav, panel, pool, config):
        self.bond_nav = bond_nav
        self.panel = panel
        self.pool = pool
        self.config = config

    def run(self, freq=None):
        # 简化: 80% 债券 + 20% 动量
        bond = self.bond_nav
        if freq:
            self.config.rebalance_freq = freq
        # 占位, 实际不跑
        return FixedIncomePlusResult(nav=bond)


__all__ = [
    "FixedIncomePlus",
    "FixedIncomePlusConfig",
    "FixedIncomePlusResult",
]
