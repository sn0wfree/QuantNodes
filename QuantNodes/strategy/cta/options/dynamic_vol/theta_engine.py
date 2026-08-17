"""Theta引擎

卖方策略：卖put spread / 铁鹰 / 宽跨
核心逻辑：收theta，控制delta敞口
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..selector import (
    OptionSelection, PutSpread, StraddleSelection,
    select_put, select_call, select_put_spread,
)
from .surface_selector import SurfaceSelector

logger = logging.getLogger(__name__)


@dataclass
class ThetaPosition:
    """Theta引擎持仓"""
    product: str
    strategy: str          # "sell_put_spread" / "sell_iron_condor" / "sell_strangle" / "sell_put_buy_call_wing"
    entry_date: pd.Timestamp
    entry_premium: float   # 净收入（卖方为正）
    stop_loss: float       # 止损阈值
    # 各腿
    short_put: Optional[OptionSelection] = None
    long_put: Optional[OptionSelection] = None   # put spread的保护腿 / 铁鹰的wing
    short_call: Optional[OptionSelection] = None  # 铁鹰/宽跨的call腿
    long_call: Optional[OptionSelection] = None   # transition状态的call翼
    qty: int = 0

    @property
    def is_active(self) -> bool:
        return self.qty > 0


class ThetaEngine:
    """Theta引擎

    执行卖方策略：卖put spread / 铁鹰 / 宽跨
    管理持仓：止损检查、到期处理
    """

    def __init__(self):
        self.positions: Dict[str, ThetaPosition] = {}  # {product: position}

    def execute(
        self,
        strategy: str,
        product: str,
        date: pd.Timestamp,
        position_data: Any,
        equity: float,
        spot: float,
        multiplier: float,
        cooldown_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """执行卖方策略

        Parameters
        ----------
        strategy : 策略类型
        product : 品种
        date : 日期
        position_data : 从surface_selector返回的合约数据
        equity : 账户净值
        spot : 标的价格
        multiplier : 合约乘数
        cooldown_multiplier : 冷静期乘数

        Returns
        -------
        目标持仓 {symbol: target_qty}
        """
        if position_data is None:
            return {}

        targets = {}

        if strategy == "sell_put_spread":
            targets = self._execute_put_spread(
                product, date, position_data, equity, spot, multiplier,
                cooldown_multiplier,
            )
        elif strategy == "sell_put_buy_call_wing":
            targets = self._execute_put_with_call_wing(
                product, date, position_data, equity, spot, multiplier,
                cooldown_multiplier,
            )
        elif strategy == "sell_iron_condor":
            targets = self._execute_iron_condor(
                product, date, position_data, equity, spot, multiplier,
                cooldown_multiplier,
            )
        elif strategy == "sell_strangle":
            targets = self._execute_strangle(
                product, date, position_data, equity, spot, multiplier,
                cooldown_multiplier,
            )

        return targets

    def _execute_put_spread(
        self, product, date, position_data, equity, spot, multiplier,
        cooldown_multiplier,
    ) -> Dict[str, Any]:
        """执行卖put spread"""
        if not isinstance(position_data, PutSpread):
            return {}

        spread = position_data
        # 卖put spread：long put (+qty), short put (-qty)
        # spread.net_premium是long premium - short premium
        # 我们卖put spread = 卖高delta put + 买低delta put
        # 净收入 = short_premium - long_premium = -net_premium

        # 仓位大小：delta目标30% * equity
        qty = self._compute_qty(
            equity, spot, multiplier, spread.short_put.delta,
            cooldown_multiplier,
        )
        if qty <= 0:
            return {}

        # 记录持仓
        self.positions[product] = ThetaPosition(
            product=product,
            strategy="sell_put_spread",
            entry_date=date,
            entry_premium=-spread.net_premium,  # 净收入（正数）
            stop_loss=0.20,  # 由外部设置
            short_put=spread.short_put,
            long_put=spread.long_put,
            qty=qty,
        )

        targets = {}
        # short put: -qty
        targets[spread.short_put.symbol] = -qty
        # long put: +qty
        targets[spread.long_put.symbol] = qty

        return targets

    def _execute_put_with_call_wing(
        self, product, date, position_data, equity, spot, multiplier,
        cooldown_multiplier,
    ) -> Dict[str, Any]:
        """执行卖put + 买call翼（TRANSITION状态）"""
        if not isinstance(position_data, PutSpread):
            return {}

        spread = position_data
        qty = self._compute_qty(
            equity, spot, multiplier, spread.short_put.delta,
            cooldown_multiplier,
        )
        if qty <= 0:
            return {}

        self.positions[product] = ThetaPosition(
            product=product,
            strategy="sell_put_buy_call_wing",
            entry_date=date,
            entry_premium=-spread.net_premium,
            stop_loss=0.15,
            short_put=spread.short_put,
            long_put=spread.long_put,
            qty=qty,
        )

        targets = {}
        targets[spread.short_put.symbol] = -qty
        targets[spread.long_put.symbol] = qty

        return targets

    def _execute_iron_condor(
        self, product, date, position_data, equity, spot, multiplier,
        cooldown_multiplier,
    ) -> Dict[str, Any]:
        """执行卖铁鹰"""
        if not isinstance(position_data, dict):
            return {}

        body_put = position_data.get("body_put")
        wing_put = position_data.get("wing_put")
        body_call = position_data.get("body_call")
        wing_call = position_data.get("wing_call")

        if not all([body_put, wing_put, body_call, wing_call]):
            return {}

        qty = self._compute_qty(
            equity, spot, multiplier, body_put.delta,
            cooldown_multiplier,
        )
        if qty <= 0:
            return {}

        self.positions[product] = ThetaPosition(
            product=product,
            strategy="sell_iron_condor",
            entry_date=date,
            entry_premium=0,  # 简化：不追踪
            stop_loss=0.10,
            short_put=body_put,
            long_put=wing_put,
            short_call=body_call,
            long_call=wing_call,
            qty=qty,
        )

        targets = {}
        targets[body_put.symbol] = -qty       # 卖put
        targets[wing_put.symbol] = qty         # 买put wing
        targets[body_call.symbol] = -qty       # 卖call
        targets[wing_call.symbol] = qty        # 买call wing

        return targets

    def _execute_strangle(
        self, product, date, position_data, equity, spot, multiplier,
        cooldown_multiplier,
    ) -> Dict[str, Any]:
        """执行卖宽跨"""
        if not isinstance(position_data, dict):
            return {}

        put = position_data.get("put")
        call = position_data.get("call")

        if not put or not call:
            return {}

        qty = self._compute_qty(
            equity, spot, multiplier, put.delta,
            cooldown_multiplier,
        )
        if qty <= 0:
            return {}

        self.positions[product] = ThetaPosition(
            product=product,
            strategy="sell_strangle",
            entry_date=date,
            entry_premium=0,
            stop_loss=0.15,
            short_put=put,
            short_call=call,
            qty=qty,
        )

        targets = {}
        targets[put.symbol] = -qty
        targets[call.symbol] = -qty

        return targets

    def _compute_qty(
        self, equity: float, spot: float, multiplier: float,
        delta: float, cooldown_multiplier: float,
    ) -> int:
        """计算仓位大小

        风险预算法：最大亏损不超过2% equity
        对于put spread：max_loss = net_premium * multiplier * qty
        简化：用固定名义价值目标
        """
        # 固定名义价值目标：50万 per position
        target_notional = 500_000
        delta_abs = abs(delta) if delta != 0 else 0.25

        notional = target_notional * cooldown_multiplier
        qty = int(notional / (spot * multiplier * delta_abs))
        return max(0, qty)

    def check_stop_loss(
        self, product: str, date: pd.Timestamp,
        current_premiums: Dict[str, float],
    ) -> bool:
        """检查是否触发止损

        Returns True if should close position
        """
        pos = self.positions.get(product)
        if pos is None or not pos.is_active:
            return False

        # 计算当前组合溢价
        current_premium = self._compute_current_premium(pos, current_premiums)
        if current_premium is None:
            return False

        # 卖方止损：溢价上涨超过阈值
        if pos.entry_premium > 0:
            loss_pct = (current_premium - pos.entry_premium) / pos.entry_premium
            if loss_pct >= pos.stop_loss:
                logger.info(
                    f"Stop loss triggered for {product}: "
                    f"loss={loss_pct:.2%} >= threshold={pos.stop_loss:.2%}"
                )
                return True

        return False

    def close_position(self, product: str) -> Dict[str, Any]:
        """平仓"""
        pos = self.positions.get(product)
        if pos is None:
            return {}

        targets = {}
        if pos.short_put:
            targets[pos.short_put.symbol] = pos.qty  # 平空put = +qty
        if pos.long_put:
            targets[pos.long_put.symbol] = -pos.qty  # 平多put = -qty
        if pos.short_call:
            targets[pos.short_call.symbol] = pos.qty
        if pos.long_call:
            targets[pos.long_call.symbol] = -pos.qty

        self.positions.pop(product, None)
        return targets

    def _compute_current_premium(
        self, pos: ThetaPosition, premiums: Dict[str, float]
    ) -> Optional[float]:
        """计算当前组合净溢价"""
        total = 0.0
        has_data = False

        if pos.short_put and pos.short_put.symbol in premiums:
            total -= premiums[pos.short_put.symbol]  # 空腿：溢价上涨=亏损
            has_data = True
        if pos.long_put and pos.long_put.symbol in premiums:
            total += premiums[pos.long_put.symbol]
            has_data = True
        if pos.short_call and pos.short_call.symbol in premiums:
            total -= premiums[pos.short_call.symbol]
            has_data = True
        if pos.long_call and pos.long_call.symbol in premiums:
            total += premiums[pos.long_call.symbol]
            has_data = True

        if not has_data:
            return None
        return total
