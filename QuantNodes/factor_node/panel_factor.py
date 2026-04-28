# coding=utf-8
"""
PanelFactorNode - 面板运算因子节点

结合时间序列和截面运算的面板数据处理。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from QuantNodes.factor_node.factor_node import FactorNode


class PanelFactorNode(FactorNode):
    """
    面板运算因子节点

    结合时间序列和截面运算，如日间因子 -> 截面 rank -> 日间平均。

    Examples:
        >>> # 因子去行业中性和市值中性
        >>> factor = PanelFactorNode(
        ...     operations=[
        ...         ("zscore", {"column": "return", "groupby": "industry"}),
        ...         ("zscore", {"column": "return", "groupby": "size"}),
        ...     ],
        ...     combine="add"
        ... )
        >>> result = factor.execute(df)
    """

    def __init__(
        self,
        operations: Optional[List[tuple]] = None,
        combine: str = "add",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            operations: 操作列表，每项为 (operation_name, params) 元组
            combine: 组合方式 "add", "mul", "mean"
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or "PanelFactor", config=config, **kwargs)
        self.operations = operations or []
        self.combine = combine

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行面板运算"""
        if input_data is None:
            raise ValueError("input_data is required for PanelFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()
        results = []

        for op_name, params in self.operations:
            if op_name == "zscore":
                column = params.get("column")
                groupby = params.get("groupby")
                result = self._zscore(df, column, groupby)
                results.append(result)

            elif op_name == "demean":
                column = params.get("column")
                groupby = params.get("groupby")
                result = self._demean(df, column, groupby)
                results.append(result)

            elif op_name == "rank":
                column = params.get("column")
                groupby = params.get("groupby")
                pct = params.get("pct", True)
                result = self._rank(df, column, groupby, pct)
                results.append(result)

            else:
                raise ValueError(f"Unknown operation: {op_name}")

        if not results:
            raise ValueError("No operations specified")

        return self._combine_results(results)

    def _zscore(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None) -> pd.Series:
        """Z-score 标准化"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

    def _demean(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None) -> pd.Series:
        """去均值"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].transform(
                lambda x: x - x.mean()
            )
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].transform(
                lambda x: x - x.mean()
            )

    def _rank(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None, pct: bool = True) -> pd.Series:
        """排名"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].rank(pct=pct)
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].rank(pct=pct)

    def _combine_results(self, results: List[pd.Series]) -> pd.DataFrame:
        """合并多个结果"""
        if self.combine == "add":
            combined = sum(results)
        elif self.combine == "mul":
            combined = results[0]
            for r in results[1:]:
                combined = combined * r
        elif self.combine == "mean":
            combined = pd.concat(results, axis=1).mean(axis=1)
        else:
            raise ValueError(f"Unknown combine method: {self.combine}")

        combined.name = self.name
        return combined.to_frame()

    def __repr__(self) -> str:
        return f"<PanelFactorNode operations={len(self.operations)}>"


class DelayFactorNode(FactorNode):
    """
    延迟因子节点

    返回过去 N 天的因子值。
    """

    def __init__(
        self,
        base_factor: FactorNode,
        periods: int = 1,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(Name=name or f"Delay{periods}", config=config, **kwargs)
        self.base_factor = base_factor
        self.periods = periods

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行延迟操作"""
        if input_data is None:
            raise ValueError("input_data is required for DelayFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        
        if isinstance(base_result, pd.DataFrame):
            col = base_result.iloc[:, 0]
        else:
            col = base_result
            
        result = col.shift(self.periods)
        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<DelayFactorNode periods={self.periods}>"


class DeltaFactorNode(FactorNode):
    """
    变化率因子节点

    计算因子值的变化率或差值。
    """

    def __init__(
        self,
        base_factor: FactorNode,
        periods: int = 1,
        mode: str = "diff",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            base_factor: 基础因子
            periods: 滞后期数
            mode: "diff" (差分) 或 "pct_change" (百分比变化)
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"Delta{mode.capitalize()}", config=config, **kwargs)
        self.base_factor = base_factor
        self.periods = periods
        self.mode = mode

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行差分/变化率操作"""
        if input_data is None:
            raise ValueError("input_data is required for DeltaFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        
        if isinstance(base_result, pd.DataFrame):
            col = base_result.iloc[:, 0]
        else:
            col = base_result

        if self.mode == "diff":
            result = col.diff(self.periods)
        elif self.mode == "pct_change":
            result = col.pct_change(self.periods)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<DeltaFactorNode periods={self.periods} mode='{self.mode}'>"
