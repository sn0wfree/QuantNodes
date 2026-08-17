"""希腊字母对冲管理器

Delta对冲：通过期货对冲组合delta敞口
Vega监控：总vega不超过NAV的1%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HedgerConfig:
    """对冲配置"""
    delta_hedge_threshold: float = 0.10  # delta敞口>10% NAV时对冲
    vega_limit: float = 0.01             # 总vega不超过NAV的1%
    min_hedge_qty: int = 1               # 最小对冲量


class GreeksHedger:
    """希腊字母对冲管理器

    - Delta对冲：组合|delta| > 10% NAV时，通过期货对冲
    - Vega监控：总|vega|不超过NAV的1%
    """

    def __init__(self, config: Optional[HedgerConfig] = None):
        self.config = config or HedgerConfig()

    def compute_delta_hedge(
        self,
        portfolio_delta: float,
        equity: float,
        spot: float,
        multiplier: float,
    ) -> int:
        """计算Delta对冲量（期货手数）

        Parameters
        ----------
        portfolio_delta : 组合总delta（以合约为单位）
        equity : 账户净值
        spot : 标的价格
        multiplier : 合约乘数

        Returns
        -------
        对冲期货手数（正=买，负=卖，0=不需要对冲）
        """
        # delta敞口（名义价值）
        delta_notional = abs(portfolio_delta) * spot * multiplier
        delta_pct = delta_notional / equity if equity > 0 else 0

        if delta_pct <= self.config.delta_hedge_threshold:
            return 0  # 不需要对冲

        # 计算对冲量
        hedge_qty = int(delta_notional / (spot * multiplier))
        if hedge_qty < self.config.min_hedge_qty:
            return 0

        # 方向：与组合delta相反
        if portfolio_delta > 0:
            return -hedge_qty  # 组合偏多 → 卖期货对冲
        else:
            return hedge_qty   # 组合偏空 → 买期货对冲

    def check_vega_limit(
        self, portfolio_vega: float, equity: float
    ) -> bool:
        """检查Vega敞口是否超限

        Returns True if within limit
        """
        if equity <= 0:
            return True
        vega_pct = abs(portfolio_vega) / equity
        return vega_pct <= self.config.vega_limit

    def aggregate_delta(
        self, positions: Dict[str, float], deltas: Dict[str, float]
    ) -> float:
        """聚合组合delta

        Parameters
        ----------
        positions : {symbol: qty} 各合约持仓
        deltas : {symbol: delta} 各合约delta

        Returns
        -------
        组合总delta
        """
        total = 0.0
        for sym, qty in positions.items():
            delta = deltas.get(sym, 0.0)
            total += qty * delta
        return total

    def aggregate_vega(
        self, positions: Dict[str, float], vegas: Dict[str, float]
    ) -> float:
        """聚合组合vega"""
        total = 0.0
        for sym, qty in positions.items():
            vega = vegas.get(sym, 0.0)
            total += qty * vega
        return total
