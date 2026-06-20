# coding=utf-8
"""
DisplayNode - 数据可视化节点模块

提供通用数据可视化节点，将处理后的数据转换为前端可用的格式。
支持表格、图表、指标、文本等可视化类型。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from QuantNodes.core.node import BaseNode


class VisualizationType(str, Enum):
    """可视化类型枚举"""
    TABLE = "table"
    CHART = "chart"
    METRIC = "metric"
    TEXT = "text"
    IMAGE = "image"


@dataclass
class VisualizationData:
    """可视化数据容器"""
    viz_type: VisualizationType = VisualizationType.TABLE
    title: str = ""
    data: Any = None
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DisplayNode(BaseNode[Any, VisualizationData], ABC):
    """
    数据可视化节点基类

    提供统一的数据可视化接口，将处理后的数据转换为可视化格式。

    Subclasses must implement:
        _prepare_data(): 准备可视化数据
    """

    _enable_validation: bool = False
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or self.__class__.__name__, config=config, **kwargs)
        self._viz_data: Optional[VisualizationData] = None

    @abstractmethod
    def _prepare_data(self, input_data: Any, **kwargs) -> VisualizationData:
        """
        准备可视化数据

        Args:
            input_data: 输入数据
            **kwargs: 额外参数

        Returns:
            可视化数据
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> VisualizationData:
        """
        执行可视化数据准备

        Args:
            input_data: 输入数据
            **kwargs: 额外参数

        Returns:
            可视化数据
        """
        self._viz_data = self._prepare_data(input_data, **kwargs)
        return self._viz_data


class TableDisplayNode(DisplayNode):
    """
    表格显示节点

    将 DataFrame 或类表格数据转换为表格可视化格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        **kwargs
    ):
        super().__init__(name=name or "TableDisplay", config=config, **kwargs)
        self.title = title

    def _prepare_data(self, input_data: Any, **kwargs) -> VisualizationData:
        """准备表格数据"""
        if isinstance(input_data, pd.DataFrame):
            data = input_data
            columns = list(input_data.columns)
        elif isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
            columns = list(input_data.keys())
        else:
            data = input_data
            columns = []

        return VisualizationData(
            viz_type=VisualizationType.TABLE,
            title=self.title,
            data=data,
            columns=columns,
            metadata={}
        )


class ChartDisplayNode(DisplayNode):
    """
    图表显示节点

    将数据转换为图表可视化格式（Line/Bar/Area/Pie）。
    """

    CHART_TYPES = ["line", "bar", "area", "pie", "scatter"]

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        chart_type: str = "line",
        **kwargs
    ):
        super().__init__(name=name or "ChartDisplay", config=config, **kwargs)
        self.title = title
        self.chart_type = chart_type if chart_type in self.CHART_TYPES else "line"

    def _prepare_data(self, input_data: Any, **kwargs) -> VisualizationData:
        """准备图表数据"""
        if isinstance(input_data, pd.DataFrame):
            data = input_data
        elif isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
        else:
            data = input_data

        return VisualizationData(
            viz_type=VisualizationType.CHART,
            title=self.title,
            data=data,
            metadata={"chart_type": self.chart_type}
        )


class MetricDisplayNode(DisplayNode):
    """
    指标显示节点

    将单个或多个指标值转换为指标可视化格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        **kwargs
    ):
        super().__init__(name=name or "MetricDisplay", config=config, **kwargs)
        self.title = title

    def _prepare_data(self, input_data: Any, **kwargs) -> VisualizationData:
        """准备指标数据"""
        value = input_data
        metadata = {}

        if isinstance(input_data, dict):
            value = input_data.get('value', input_data)
            if 'delta' in input_data:
                metadata['delta'] = input_data['delta']
            if 'description' in input_data:
                metadata['description'] = input_data['description']

        return VisualizationData(
            viz_type=VisualizationType.METRIC,
            title=self.title,
            data=value,
            metadata=metadata
        )


class TextDisplayNode(DisplayNode):
    """
    文本显示节点

    将文本内容转换为文本可视化格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        **kwargs
    ):
        super().__init__(name=name or "TextDisplay", config=config, **kwargs)
        self.title = title

    def _prepare_data(self, input_data: Any, **kwargs) -> VisualizationData:
        """准备文本数据"""
        return VisualizationData(
            viz_type=VisualizationType.TEXT,
            title=self.title,
            data=input_data,
            metadata={}
        )
