# coding=utf-8
"""
BrokerNode - 经纪商节点

提供经纪商节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from QuantNodes.core.node import BaseNode
from QuantNodes.backtest.strategy_node import Order, OrdersResult


@dataclass
class Trade:
    """成交记录数据结构"""
    order_id: str
    code: str
    side: str
    size: float
    price: float
    adjusted_price: float
    fee: float
    dt: str
    status: str = 'completed'


@dataclass
class TradeResult:
    """成交结果容器"""
    trades: List[Trade] = field(default_factory=list)
    cash: float = 0.0
    commission: float = 0.0
    executed_value: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                'order_id': t.order_id,
                'code': t.code,
                'side': t.side,
                'size': t.size,
                'price': t.price,
                'adjusted_price': t.adjusted_price,
                'fee': t.fee,
                'dt': t.dt,
                'status': t.status,
            }
            for t in self.trades
        ])


class BrokerNode(BaseNode[OrdersResult, TradeResult], ABC):
    """
    经纪商节点基类

    提供统一的订单执行接口。

    Subclasses must implement:
        _execute_orders(): 执行具体订单逻辑

    Examples:
        >>> broker = BrokerNode(config={
        ...     'cash': 100000,
        ...     'commission': 0.001,
        ...     'margin': 0.1
        ... })
        >>> result = broker.execute(orders)
    """

    _enable_validation: bool = True
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)

        self._result: Optional[TradeResult] = None
        self._cash: float = self.config.get('cash', 100000)
        self._commission: float = self.config.get('commission', 0.001)
        self._margin: float = self.config.get('margin', 0.1)
        self._leverage: float = 1 / self._margin if self._margin > 0 else 1
        self._trade_on_close: bool = self.config.get('trade_on_close', False)
        self._hedging: bool = self.config.get('hedging', False)

        self._positions: Dict[str, float] = {}
        self._initial_cash: float = self._cash

    @abstractmethod
    def _execute_orders(
        self,
        orders: OrdersResult,
        quote_data: pd.DataFrame,
        **kwargs
    ) -> TradeResult:
        """
        执行具体订单逻辑

        Args:
            orders: 订单结果
            quote_data: 行情数据
            **kwargs: 额外执行参数

        Returns:
            TradeResult 成交结果
        """
        pass

    def _execute(
        self,
        input_data: Union[OrdersResult, tuple, None] = None,
        **kwargs
    ) -> TradeResult:
        """
        执行订单

        Args:
            input_data: 输入数据，可以是：
                - OrdersResult: 订单结果
                - tuple: (orders, quote_data) 元组
                - None: 使用 config 中的默认数据
            **kwargs: 额外执行参数

        Returns:
            TradeResult 成交结果
        """
        if input_data is None:
            orders = kwargs.get('orders')
            quote_data = kwargs.get('quote_data')
        elif isinstance(input_data, OrdersResult):
            orders = input_data
            quote_data = kwargs.get('quote_data')
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            orders, quote_data = input_data
        else:
            raise ValueError(
                f"input_data must be OrdersResult or (orders, quote_data) tuple, "
                f"got {type(input_data).__name__}"
            )

        if orders is None:
            raise ValueError("orders is required")

        self._result = self._execute_orders(orders, quote_data, **kwargs)
        return self._result

    def _validate_input(self, input_data: Any) -> None:
        """验证输入数据"""
        if input_data is None:
            return

        if isinstance(input_data, OrdersResult):
            pass
        elif isinstance(input_data, tuple):
            orders, quote_data = input_data
            if not isinstance(orders, OrdersResult):
                raise ValueError(f"orders must be OrdersResult, got {type(orders).__name__}")
        else:
            raise ValueError(
                f"input_data must be OrdersResult or tuple, got {type(input_data).__name__}"
            )

    def get_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        return self._positions.copy()

    def get_cash(self) -> float:
        """获取当前现金"""
        return self._cash

    def reset(self) -> None:
        """重置经纪商状态"""
        super().reset()
        self._cash = self.config.get('cash', 100000)
        self._positions = {}
        self._result = None


class SimulatedBrokerNode(BrokerNode):
    """模拟经纪商节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "SimulatedBroker", config=config, **kwargs)

    def _execute_orders(
        self,
        orders: OrdersResult,
        quote_data: pd.DataFrame,
        **kwargs
    ) -> TradeResult:
        """模拟执行订单"""
        result = TradeResult()
        result.cash = self._cash

        if quote_data is None or quote_data.empty:
            return result

        quote_dict = {}
        if 'Code' in quote_data.columns and 'date' in quote_data.columns:
            for _, row in quote_data.iterrows():
                key = (row['Code'], row['date'])
                quote_dict[key] = row

        for order in orders.orders:
            order_id = order.order_id or f"order_{len(result.trades)}"
            dt = order.create_date

            price_col = 'Close' if self._trade_on_close else 'Open'
            key = (order.code, dt)

            if key in quote_dict:
                quote_row = quote_dict[key]
                price = quote_row.get(price_col, quote_row.get('Close'))
            else:
                code_df = quote_data[
                    (quote_data['Code'] == order.code)
                ]
                if code_df.empty:
                    continue
                price = code_df.iloc[-1].get(price_col, code_df.iloc[-1].get('Close'))

            adjusted_price = self._adjusted_price(price, order.size)
            fee = abs(order.size * adjusted_price * self._commission)

            if order.size > 0:
                cost = order.size * adjusted_price + fee
            else:
                cost = -fee

            if result.cash >= cost or order.size < 0:
                result.cash -= cost

                current_pos = self._positions.get(order.code, 0)
                self._positions[order.code] = current_pos + order.size

                trade = Trade(
                    order_id=order_id,
                    code=order.code,
                    side='buy' if order.size > 0 else 'sell',
                    size=abs(order.size),
                    price=price,
                    adjusted_price=adjusted_price,
                    fee=fee,
                    dt=dt,
                )
                result.trades.append(trade)
                result.commission += fee

        result.cash = self._cash
        return result

    def _adjusted_price(self, price: float, size: float) -> float:
        """调整价格（含手续费）"""
        return price * (1 + np.copysign(self._commission, size))


class ExecutionBrokerNode(BrokerNode):
    """执行经纪商节点（更真实的订单执行）"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "ExecutionBroker", config=config, **kwargs)
        self._slippage = self.config.get('slippage', 0.0005)

    def _execute_orders(
        self,
        orders: OrdersResult,
        quote_data: pd.DataFrame,
        **kwargs
    ) -> TradeResult:
        """执行订单（含滑点）- 向量化实现"""
        result = TradeResult()
        result.cash = self._cash

        if quote_data is None or quote_data.empty or not orders.orders:
            return result

        price_col = 'Close' if self._trade_on_close else 'Open'
        code_col = 'Code' if 'Code' in quote_data.columns else 'code'
        date_col = 'date' if 'date' in quote_data.columns else 'Date'

        # 用 numpy 向量化构建查找表 (比 iterrows 快 10-50x)
        q_codes = quote_data[code_col].astype(str).values
        q_dates = quote_data[date_col].astype(str).values
        q_close = quote_data['Close'].values if 'Close' in quote_data.columns else None
        q_open = quote_data[price_col].values if price_col in quote_data.columns else q_close

        # 构建 {(code, date): (open, close)} 查找表
        price_map: Dict[str, Dict[str, tuple]] = {}
        for i in range(len(q_codes)):
            c, d = q_codes[i], q_dates[i]
            if c not in price_map:
                price_map[c] = {}
            price_map[c][d] = (float(q_open[i]), float(q_close[i]) if q_close is not None else float(q_open[i]))

        # 构建每个 code 的首条/末条 fallback
        code_first: Dict[str, tuple] = {}
        code_last: Dict[str, tuple] = {}
        for c, dmap in price_map.items():
            if dmap:
                first_key = next(iter(dmap))
                code_first[c] = dmap[first_key]
                last_key = next(reversed(dmap))
                code_last[c] = dmap[last_key]

        # 向量化处理订单
        n_orders = len(orders.orders)
        order_codes = [o.code for o in orders.orders]
        order_sizes = np.array([o.size for o in orders.orders])
        order_dates = [str(o.create_date) if o.create_date is not None else None for o in orders.orders]
        order_ids = [o.order_id or f"order_{i}" for i, o in enumerate(orders.orders)]

        # 向量化价格查找
        prices = np.empty(n_orders)
        for i in range(n_orders):
            c = order_codes[i]
            d = order_dates[i]
            dmap = price_map.get(c)
            if dmap is None:
                prices[i] = 0.0
                continue
            if d is not None:
                row = dmap.get(d)
                if row is None:
                    row = code_first.get(c)
            else:
                row = code_last.get(c)
            prices[i] = row[0] if row else 0.0  # Open price (index 0)

        # 向量化滑点和费用计算
        slippage_factors = 1.0 + np.copysign(self._slippage, order_sizes)
        commission_factors = 1.0 + np.copysign(self._commission, order_sizes)
        adjusted_prices = prices * slippage_factors * commission_factors
        fees = np.abs(order_sizes * adjusted_prices * self._commission)
        costs = np.where(order_sizes > 0, order_sizes * adjusted_prices + fees, -fees)

        # 逐笔执行（顺序执行因为有现金依赖）
        cash = self._cash
        for i in range(n_orders):
            if prices[i] == 0.0:
                continue
            if cash >= costs[i] or order_sizes[i] < 0:
                cash -= costs[i]
                code = order_codes[i]
                self._positions[code] = self._positions.get(code, 0.0) + order_sizes[i]

                trade = Trade(
                    order_id=order_ids[i],
                    code=code,
                    side='buy' if order_sizes[i] > 0 else 'sell',
                    size=abs(order_sizes[i]),
                    price=float(prices[i]),
                    adjusted_price=float(adjusted_prices[i]),
                    fee=float(fees[i]),
                    dt=order_dates[i],
                )
                result.trades.append(trade)
                result.commission += fees[i]

        result.cash = cash
        return result

    def _adjusted_price(self, price: float, size: float) -> float:
        """调整价格（含手续费和滑点）"""
        return price * (1 + np.copysign(self._commission + self._slippage, size))