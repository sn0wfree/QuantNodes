# coding=utf-8
"""
FactorNode - 因子计算节点基类

提供因子计算节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from QuantNodes.core.node import BaseNode


class FactorNode(BaseNode, ABC):
    """
    因子计算节点基类

    所有因子计算节点都继承自此类，提供统一的因子计算接口。

    Subclasses must implement:
        _compute(): 执行具体因子计算

    Examples:
        >>> # 单点运算
        >>> factor = PointFactorNode(expression="close / open - 1")
        >>> result = factor.execute(data)
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)
        self._result: Optional[pd.DataFrame] = None

    @abstractmethod
    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """
        执行因子计算

        Args:
            input_data: 输入数据（DataFrame 或数据库连接）
            **kwargs: 额外执行参数

        Returns:
            因子计算结果 DataFrame
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行因子计算"""
        self._result = self._compute(input_data, **kwargs)
        return self._result

    def __rshift__(self, other: 'FactorNode') -> 'FactorPipeline':
        """重载 >> 运算符用于因子组合"""
        return FactorPipeline([self, other])

    def then(self, other: 'FactorNode') -> 'FactorPipeline':
        """链式调用"""
        return FactorPipeline([self, other])


class FactorPipeline:
    """
    因子管道

    将多个因子节点组合在一起计算。
    """

    def __init__(self, factors: List[FactorNode]):
        self.factors = factors

    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """执行所有因子计算"""
        results = {}
        for i, factor in enumerate(self.factors):
            name = factor.name if hasattr(factor, 'name') else f"Factor{i}"
            if name in results:
                name = f"{name}_{i}"
            results[name] = factor.execute(input_data, **kwargs)
        return results

    def __rshift__(self, other: FactorNode) -> 'FactorPipeline':
        """重载 >> 运算符"""
        if isinstance(other, FactorPipeline):
            return FactorPipeline(self.factors + other.factors)
        return FactorPipeline(self.factors + [other])
