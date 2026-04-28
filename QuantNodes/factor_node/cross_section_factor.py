# coding=utf-8
"""
CrossSectionFactorNode - 截面运算因子节点

对某一时间点的所有标的进行横截面运算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from QuantNodes.factor_node.factor_node import FactorNode


class CrossSectionFactorNode(FactorNode):
    """
    截面运算因子节点

    对某一时间点的所有标的进行横截面运算，如市值加权、排名等。

    Examples:
        >>> # 横截面排名
        >>> factor = CrossSectionFactorNode(
        ...     operation="rank",
        ...     column="return"
        ... )
        >>> result = factor.execute(df)
        >>>
        >>> # 横截面去均值
        >>> factor = CrossSectionFactorNode(
        ...     operation="demean",
        ...     column="return"
        ... )
    """

    def __init__(
        self,
        operation: str = "rank",
        column: Optional[str] = None,
        groupby: Optional[str] = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            operation: 运算类型 "rank", "zscore", "demean", "mad", "percentile"
            column: 要操作的列名
            groupby: 分组列名（如行业分组）
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"CrossSection{operation.capitalize()}", config=config, **kwargs)
        self.operation = operation
        self.column = column
        self.groupby = groupby

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行截面运算"""
        if input_data is None:
            raise ValueError("input_data is required for CrossSectionFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if self.column is None:
            raise ValueError("column is required for CrossSectionFactorNode")

        if self.operation == "rank":
            result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].rank(
                pct=True if 'pct' in str(kwargs.get('mode', '')) else False
            )

        elif self.operation == "zscore":
            def zscore(x):
                return (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
            
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(zscore)
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(zscore)

        elif self.operation == "demean":
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(
                    lambda x: x - x.mean()
                )
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(
                    lambda x: x - x.mean()
                )

        elif self.operation == "mad":
            def mad(x):
                return (x - x.median()).abs().median() * 1.4826
            
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(mad)
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(mad)

        elif self.operation == "percentile":
            result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].rank(pct=True)

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<CrossSectionFactorNode operation='{self.operation}' column='{self.column}'>"


class GroupRankFactorNode(FactorNode):
    """
    分组排名因子节点

    按组别进行排名，常用于 Barra 风格因子。
    """

    def __init__(
        self,
        column: str,
        groupby: str,
        ascending: bool = False,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(Name=name or "GroupRank", config=config, **kwargs)
        self.column = column
        self.groupby = groupby
        self.ascending = ascending

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行分组排名"""
        if input_data is None:
            raise ValueError("input_data is required for GroupRankFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if 'dt' in df.columns:
            result = df.groupby(['dt', self.groupby])[self.column].rank(ascending=self.ascending)
        else:
            result = df.groupby(self.groupby)[self.column].rank(ascending=self.ascending)

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<GroupRankFactorNode column='{self.column}' groupby='{self.groupby}'>"
