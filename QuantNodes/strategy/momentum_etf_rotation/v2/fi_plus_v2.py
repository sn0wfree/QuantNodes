# coding=utf-8
"""固收+ (80/20) 模块 (Stage 11 精简 stub)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
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


def performance_metrics(nav: pd.Series, freq: int = 252) -> dict:
    """标准业绩指标 (与 portfolio.py 中一致)."""
    if nav.empty or len(nav) < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0}
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(rets) / freq
    ann_return = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(rets.std() * np.sqrt(freq))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": max_dd, "calmar": calmar,
    }


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