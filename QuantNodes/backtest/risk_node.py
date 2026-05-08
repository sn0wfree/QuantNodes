# coding=utf-8
"""
RiskNode - 风控节点

提供风控节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from QuantNodes.core.node import BaseNode
from QuantNodes.backtest.strategy_node import Order, OrdersResult
from QuantNodes.backtest.broker_node import TradeResult


@dataclass
class RiskCheck:
    """风控检查结果"""
    passed: bool
    order: Order
    reason: Optional[str] = None
    adjusted_size: Optional[float] = None


@dataclass
class RiskResult:
    """风控结果容器"""
    passed_orders: List[Order] = field(default_factory=list)
    rejected_orders: List[RiskCheck] = field(default_factory=list)
    adjusted_orders: List[RiskCheck] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        if not self.passed_orders:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                'code': o.code,
                'size': o.size,
                'limit_price': o.limit_price,
                'create_date': o.create_date,
            }
            for o in self.passed_orders
        ])


class RiskNode(BaseNode[OrdersResult, RiskResult], ABC):
    """
    风控节点基类

    提供统一的风控检查接口。

    Subclasses must implement:
        _check_order(): 检查单个订单

    Examples:
        >>> risk = RiskNode(config={'max_position': 10000})
        >>> result = risk.execute(orders)
    """

    _enable_validation: bool = True
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)

        self._result: Optional[RiskResult] = None
        self._positions: Dict[str, float] = {}
        self._max_position: float = self.config.get('max_position', float('inf'))
        self._max_order_size: float = self.config.get('max_order_size', float('inf'))
        self._min_order_size: float = self.config.get('min_order_size', 0)

    @abstractmethod
    def _check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        **kwargs
    ) -> RiskCheck:
        """
        检查单个订单

        Args:
            order: 订单
            positions: 当前持仓
            **kwargs: 额外执行参数

        Returns:
            RiskCheck 风控检查结果
        """
        pass

    def _execute(
        self,
        input_data: Union[OrdersResult, tuple, None] = None,
        **kwargs
    ) -> RiskResult:
        """
        执行风控检查

        Args:
            input_data: 输入数据，可以是：
                - OrdersResult: 订单结果
                - tuple: (orders, positions) 元组
                - None: 使用 config 中的默认数据
            **kwargs: 额外执行参数

        Returns:
            RiskResult 风控结果
        """
        if input_data is None:
            orders = kwargs.get('orders')
            positions = kwargs.get('positions', {})
        elif isinstance(input_data, OrdersResult):
            orders = input_data
            positions = kwargs.get('positions', {})
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            orders, positions = input_data
            if isinstance(positions, dict):
                pass
            elif isinstance(positions, TradeResult):
                positions = self._derive_positions_from_trades(positions)
            else:
                raise ValueError(f"positions must be dict or TradeResult, got {type(positions).__name__}")
        else:
            raise ValueError(
                f"input_data must be OrdersResult or (orders, positions) tuple, "
                f"got {type(input_data).__name__}"
            )

        if orders is None:
            raise ValueError("orders is required")

        self._result = RiskResult()

        for order in orders.orders:
            check = self._check_order(order, positions, **kwargs)
            if check.passed:
                self._result.passed_orders.append(order)
            elif check.adjusted_size is not None:
                adjusted_order = Order(
                    code=order.code,
                    size=check.adjusted_size,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    sl_price=order.sl_price,
                    tp_price=order.tp_price,
                    order_id=order.order_id,
                    create_date=order.create_date,
                )
                self._result.passed_orders.append(adjusted_order)
                self._result.adjusted_orders.append(check)
            else:
                self._result.rejected_orders.append(check)

        return self._result

    def _derive_positions_from_trades(self, trade_result: TradeResult) -> Dict[str, float]:
        """从交易结果推导持仓"""
        positions = {}
        for trade in trade_result.trades:
            current = positions.get(trade.code, 0)
            if trade.side == 'buy':
                positions[trade.code] = current + trade.size
            else:
                positions[trade.code] = current - trade.size
        return positions

    def _validate_input(self, input_data: Any) -> None:
        """验证输入数据"""
        if input_data is None:
            return

        if isinstance(input_data, OrdersResult):
            pass
        elif isinstance(input_data, tuple):
            orders, positions = input_data
            if not isinstance(orders, OrdersResult):
                raise ValueError(f"orders must be OrdersResult, got {type(orders).__name__}")
        else:
            raise ValueError(
                f"input_data must be OrdersResult or tuple, got {type(input_data).__name__}"
            )

    def get_passed_orders(self) -> List[Order]:
        """获取通过的订单"""
        if self._result is None:
            return []
        return self._result.passed_orders

    def get_rejected_orders(self) -> List[RiskCheck]:
        """获取拒绝的订单"""
        if self._result is None:
            return []
        return self._result.rejected_orders

    def set_positions(self, positions: Dict[str, float]) -> None:
        """设置当前持仓"""
        self._positions = positions.copy()


class PositionLimitRiskNode(RiskNode):
    """仓位限制风控节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "PositionLimitRisk", config=config, **kwargs)
        self._max_position = self.config.get('max_position', float('inf'))
        self._max_order_size = self.config.get('max_order_size', float('inf'))
        self._min_order_size = self.config.get('min_order_size', 0)

    def _check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        **kwargs
    ) -> RiskCheck:
        """检查仓位限制"""
        current_position = positions.get(order.code, 0)
        new_position = current_position + order.size

        if abs(order.size) < self._min_order_size:
            return RiskCheck(
                passed=False,
                order=order,
                reason=f"Order size {abs(order.size)} below minimum {self._min_order_size}"
            )

        if abs(order.size) > self._max_order_size:
            adjusted_size = np.copysign(self._max_order_size, order.size)
            return RiskCheck(
                passed=True,
                order=order,
                reason=f"Order size adjusted from {abs(order.size)} to {self._max_order_size}",
                adjusted_size=adjusted_size
            )

        if abs(new_position) > self._max_position:
            return RiskCheck(
                passed=False,
                order=order,
                reason=f"Position limit would be exceeded: {abs(new_position)} > {self._max_position}"
            )

        return RiskCheck(passed=True, order=order)


class StopLossRiskNode(RiskNode):
    """止损风控节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "StopLossRisk", config=config, **kwargs)
        self._max_loss = self.config.get('max_loss', float('inf'))
        self._current_pnl = 0.0

    def _check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        **kwargs
    ) -> RiskCheck:
        """检查止损风控"""
        if order.sl_price is None or order.size >= 0:
            return RiskCheck(passed=True, order=order)

        if self._current_pnl < -self._max_loss:
            return RiskCheck(
                passed=False,
                order=order,
                reason=f"Stop loss triggered: PnL {self._current_pnl} below limit {-self._max_loss}"
            )

        return RiskCheck(passed=True, order=order)

    def update_pnl(self, pnl: float) -> None:
        """更新当前盈亏"""
        self._current_pnl = pnl


class CashRiskNode(RiskNode):
    """现金风控节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "CashRisk", config=config, **kwargs)
        self._min_cash = self.config.get('min_cash', 0)

    def _check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        **kwargs
    ) -> RiskCheck:
        """检查现金限制"""
        cash = kwargs.get('cash', float('inf'))

        if order.size <= 0:
            return RiskCheck(passed=True, order=order)

        estimated_cost = order.size * (order.limit_price or 0) * (1 + self._config.get('commission', 0.001))

        if cash - estimated_cost < self._min_cash:
            return RiskCheck(
                passed=False,
                order=order,
                reason=f"Insufficient cash: need ~{estimated_cost}, have {cash}"
            )

        return RiskCheck(passed=True, order=order)


class CompositeRiskNode(RiskNode):
    """复合风控节点（组合多个风控规则）"""

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        risk_nodes: List[RiskNode] = None,
        **kwargs
    ):
        super().__init__(name=name or "CompositeRisk", config=config, **kwargs)
        self._risk_nodes = risk_nodes or []
        self._mode = self.config.get('mode', 'all')

    def add_risk_node(self, node: RiskNode) -> None:
        """添加风控节点"""
        self._risk_nodes.append(node)

    def _check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        **kwargs
    ) -> RiskCheck:
        """执行所有风控检查"""
        for risk_node in self._risk_nodes:
            check = risk_node._check_order(order, positions, **kwargs)
            if not check.passed:
                if self._mode == 'all':
                    return check
            elif check.adjusted_size is not None:
                order.size = check.adjusted_size

        if self._mode == 'any':
            return RiskCheck(passed=True, order=order)

        return RiskCheck(passed=True, order=order)