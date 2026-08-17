"""Gamma引擎

买方策略：买跨式，在波动率爆发时获利
核心逻辑：买gamma，IV crush预警时退出
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..selector import StraddleSelection
from .surface_selector import SurfaceSelector

logger = logging.getLogger(__name__)


@dataclass
class GammaPosition:
    """Gamma引擎持仓"""
    product: str
    strategy: str          # "buy_straddle"
    entry_date: pd.Timestamp
    entry_premium: float   # 买入成本（正数）
    stop_loss: float       # 止损阈值（0.50 = 50%亏损）
    straddle: Optional[StraddleSelection] = None
    qty: int = 0
    iv_at_entry: float = 0.0  # 入场时IV

    @property
    def is_active(self) -> bool:
        return self.qty > 0


class GammaEngine:
    """Gamma引擎

    执行买方策略：买跨式
    管理持仓：止损、IV crush预警退出、到期处理
    """

    def __init__(self):
        self.positions: Dict[str, GammaPosition] = {}  # {product: position}

    def execute(
        self,
        strategy: str,
        product: str,
        date: pd.Timestamp,
        position_data: Any,
        equity: float,
        spot: float,
        multiplier: float,
        iv_percentile: float = 0.5,
        cooldown_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """执行买方策略

        Parameters
        ----------
        strategy : 策略类型
        product : 品种
        date : 日期
        position_data : 从surface_selector返回的StraddleSelection
        equity : 账户净值
        spot : 标的价格
        multiplier : 合约乘数
        iv_percentile : 当前IV百分位（用于计算gamma预算）
        cooldown_multiplier : 冷静期乘数

        Returns
        -------
        目标持仓 {symbol: target_qty}
        """
        if strategy == "buy_straddle":
            return self._execute_straddle(
                product, date, position_data, equity, spot, multiplier,
                iv_percentile, cooldown_multiplier,
            )
        return {}

    def _execute_straddle(
        self, product, date, position_data, equity, spot, multiplier,
        iv_percentile, cooldown_multiplier,
    ) -> Dict[str, Any]:
        """执行买跨式"""
        if not isinstance(position_data, StraddleSelection):
            return {}

        straddle = position_data

        # Gamma预算 = equity * gamma_budget * cooldown_multiplier
        gamma_budget = self._compute_gamma_budget(
            equity, iv_percentile, cooldown_multiplier,
        )

        # 仓位大小：gamma_budget / straddle.cost / (multiplier * straddle cost per lot)
        cost_per_lot = straddle.cost * multiplier
        if cost_per_lot <= 0:
            return {}

        qty = max(1, int(gamma_budget / cost_per_lot))

        # 记录持仓
        self.positions[product] = GammaPosition(
            product=product,
            strategy="buy_straddle",
            entry_date=date,
            entry_premium=straddle.cost,
            stop_loss=0.50,  # 买方固定50%亏损止损
            straddle=straddle,
            qty=qty,
            iv_at_entry=0,  # 由外部设置
        )

        targets = {}
        # 买call: +qty
        targets[straddle.call.symbol] = qty
        # 买put: +qty
        targets[straddle.put.symbol] = qty

        logger.debug(
            f"Gamma engine: buy straddle {product} x{qty}, "
            f"cost={straddle.cost:.4f}, budget={gamma_budget:.0f}"
        )

        return targets

    def _compute_gamma_budget(
        self, equity: float, iv_percentile: float,
        cooldown_multiplier: float,
    ) -> float:
        """计算gamma引擎预算

        base_budget = 15% * equity
        iv_adjustment = max(0, 0.5 - iv_pct) * 0.5 * equity
        gamma_budget = min(25% * equity, base + adjustment)
        """
        base = 0.15
        iv_adj = max(0, 0.5 - iv_percentile) * 0.5
        gamma_pct = min(0.25, base + iv_adj)

        return equity * gamma_pct * cooldown_multiplier

    def check_stop_loss(
        self, product: str, date: pd.Timestamp,
        current_premiums: Dict[str, float],
    ) -> bool:
        """检查是否触发止损

        买方止损：溢价下跌超过50%
        """
        pos = self.positions.get(product)
        if pos is None or not pos.is_active:
            return False

        if pos.straddle is None:
            return False

        # 计算当前straddle溢价
        call_prem = current_premiums.get(pos.straddle.call.symbol)
        put_prem = current_premiums.get(pos.straddle.put.symbol)
        if call_prem is None or put_prem is None:
            return False

        current_cost = call_prem + put_prem
        loss_pct = (pos.entry_premium - current_cost) / pos.entry_premium

        if loss_pct >= pos.stop_loss:
            logger.info(
                f"Gamma stop loss triggered for {product}: "
                f"loss={loss_pct:.2%} >= threshold={pos.stop_loss:.2%}"
            )
            return True

        return False

    def check_iv_crush_exit(
        self, product: str, date: pd.Timestamp,
        iv_zscore_now: float, term_slope: float,
        iv_momentum_5d: float,
    ) -> bool:
        """检查是否因IV crush预警退出

        退出条件（任一触发）：
        1. iv_zscore < 1.0（IV回落正常）
        2. term_slope > 0（期限结构转正，backwardation→contango）
        3. iv_momentum_5d < -0.05（IV 5日下跌>5%）
        """
        pos = self.positions.get(product)
        if pos is None or not pos.is_active:
            return False

        # 条件1：IV zscore回落
        if iv_zscore_now < 1.0:
            logger.info(
                f"IV crush exit for {product}: iv_zscore={iv_zscore_now:.2f} < 1.0"
            )
            return True

        # 条件2：期限结构转正
        if term_slope > 0:
            logger.info(
                f"IV crush exit for {product}: term_slope={term_slope:.4f} > 0"
            )
            return True

        # 条件3：IV动量大幅下跌
        if iv_momentum_5d < -0.05:
            logger.info(
                f"IV crush exit for {product}: iv_mom_5d={iv_momentum_5d:.2%} < -5%"
            )
            return True

        return False

    def close_position(self, product: str) -> Dict[str, Any]:
        """平仓"""
        pos = self.positions.get(product)
        if pos is None or pos.straddle is None:
            return {}

        targets = {}
        # 卖call: -qty
        targets[pos.straddle.call.symbol] = -pos.qty
        # 卖put: -qty
        targets[pos.straddle.put.symbol] = -pos.qty

        self.positions.pop(product, None)
        return targets

    def get_position_premium(
        self, product: str, current_premiums: Dict[str, float]
    ) -> Optional[float]:
        """获取当前持仓溢价"""
        pos = self.positions.get(product)
        if pos is None or not pos.is_active or pos.straddle is None:
            return None

        call_prem = current_premiums.get(pos.straddle.call.symbol)
        put_prem = current_premiums.get(pos.straddle.put.symbol)
        if call_prem is None or put_prem is None:
            return None

        return call_prem + put_prem
