# coding=utf-8
"""
StrategyNode - 策略节点

提供策略节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from QuantNodes.core.node import BaseNode


@dataclass
class Order:
    """订单数据结构"""
    code: str
    size: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    order_id: Optional[str] = None
    create_date: Optional[str] = None


@dataclass
class Signal:
    """交易信号数据结构"""
    code: str
    signal_type: str
    strength: float = 1.0
    price: Optional[float] = None
    date: Optional[str] = None


class OrdersResult:
    """订单结果容器"""
    def __init__(self):
        self.orders: List[Order] = []
        self.signals: List[Signal] = []

    def add_order(self, order: Order) -> None:
        self.orders.append(order)

    def add_signal(self, signal: Signal) -> None:
        self.signals.append(signal)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        if not self.orders:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                'code': o.code,
                'size': o.size,
                'limit_price': o.limit_price,
                'stop_price': o.stop_price,
                'sl_price': o.sl_price,
                'tp_price': o.tp_price,
                'order_id': o.order_id,
                'create_date': o.create_date,
            }
            for o in self.orders
        ])


class StrategyNode(BaseNode[pd.DataFrame, OrdersResult], ABC):
    """
    策略节点基类

    提供统一的策略执行接口。

    Subclasses must implement:
        _generate_signals(): 生成交易信号
        _create_orders(): 根据信号创建订单

    Examples:
        >>> strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
        >>> result = strategy.execute(data)
    """

    _enable_validation: bool = True
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)

        self._result: Optional[OrdersResult] = None
        self._signals: List[Signal] = []
        self._orders: List[Order] = []

    @abstractmethod
    def _generate_signals(
        self,
        input_data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            input_data: 市场数据 DataFrame
            **kwargs: 额外执行参数

        Returns:
            信号列表
        """
        pass

    def _create_orders(
        self,
        signals: List[Signal],
        **kwargs
    ) -> List[Order]:
        """
        根据信号创建订单（默认实现）

        Args:
            signals: 信号列表
            **kwargs: 额外执行参数

        Returns:
            订单列表
        """
        orders = []
        for signal in signals:
            order = Order(
                code=signal.code,
                size=signal.strength * (1 if signal.signal_type == 'buy' else -1),
                limit_price=signal.price,
                create_date=signal.date,
            )
            orders.append(order)
        return orders

    def _execute(
        self,
        input_data: pd.DataFrame = None,
        **kwargs
    ) -> OrdersResult:
        """
        执行策略

        Args:
            input_data: 市场数据 DataFrame
            **kwargs: 额外执行参数

        Returns:
            OrdersResult 订单结果
        """
        self._signals = self._generate_signals(input_data, **kwargs)
        self._orders = self._create_orders(self._signals, **kwargs)

        result = OrdersResult()
        result.signals = self._signals
        result.orders = self._orders

        self._result = result
        return result

    def _validate_input(self, input_data: Any) -> None:
        """验证输入数据"""
        if input_data is None:
            return
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError(f"input_data must be DataFrame, got {type(input_data).__name__}")

    def get_signals(self) -> List[Signal]:
        """获取当前信号"""
        return self._signals

    def get_orders(self) -> List[Order]:
        """获取当前订单"""
        return self._orders


class MAStrategyNode(StrategyNode):
    """移动平均线策略节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "MA_Strategy", config=config, **kwargs)
        self._short_window = self.config.get('short_window', 5)
        self._long_window = self.config.get('long_window', 20)

    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]:
        """计算 MA 并生成信号"""
        if input_data is None or input_data.empty:
            return []

        df = input_data.copy()
        df['MA_Short'] = df['Close'].rolling(window=self._short_window).mean()
        df['MA_Long'] = df['Close'].rolling(window=self._long_window).mean()

        signals = []
        codes = df['Code'].unique() if 'Code' in df.columns else [None]

        for code in codes:
            if code is not None:
                code_df = df[df['Code'] == code].copy()
            else:
                code_df = df.copy()

            code_df['signal'] = 0
            code_df.loc[code_df['MA_Short'] > code_df['MA_Long'], 'signal'] = 1
            code_df.loc[code_df['MA_Short'] < code_df['MA_Long'], 'signal'] = -1

            code_df['signal_diff'] = code_df['signal'].diff()

            for _, row in code_df.iterrows():
                if row.get('signal_diff', 0) != 0:
                    signal_type = 'buy' if row['signal'] == 1 else 'sell'
                    signals.append(Signal(
                        code=code or 'DEFAULT',
                        signal_type=signal_type,
                        strength=1.0,
                        price=row.get('Close'),
                        date=str(row.get('date', '')),
                    ))

        return signals


class MomentumStrategyNode(StrategyNode):
    """动量策略节点"""

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "Momentum_Strategy", config=config, **kwargs)
        self._lookback = self.config.get('lookback', 20)
        self._threshold = self.config.get('threshold', 0.05)

    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]:
        """计算动量并生成信号"""
        if input_data is None or input_data.empty:
            return []

        df = input_data.copy()
        df['Return'] = df['Close'].pct_change(self._lookback)

        signals = []
        codes = df['Code'].unique() if 'Code' in df.columns else [None]

        for code in codes:
            if code is not None:
                code_df = df[df['Code'] == code].copy()
            else:
                code_df = df.copy()

            code_df = code_df.dropna(subset=['Return'])

            for _, row in code_df.iterrows():
                ret = row['Return']
                if abs(ret) > self._threshold:
                    signal_type = 'buy' if ret > 0 else 'sell'
                    signals.append(Signal(
                        code=code or 'DEFAULT',
                        signal_type=signal_type,
                        strength=min(abs(ret) / self._threshold, 2.0),
                        price=row.get('Close'),
                        date=str(row.get('date', '')),
                    ))

        return signals