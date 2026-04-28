# coding=utf-8
"""
UINode - UI 数据准备节点

提供 UI/Streamlit 数据准备节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from QuantNodes.core.node import BaseNode


class DisplayType(str, Enum):
    """显示类型枚举"""
    TABLE = "table"
    CHART = "chart"
    METRIC = "metric"
    TEXT = "text"
    IMAGE = "image"


@dataclass
class UIDisplayResult:
    """UI 显示结果容器"""
    display_type: DisplayType = DisplayType.TABLE
    title: str = ""
    data: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UINode(BaseNode[Any, UIDisplayResult], ABC):
    """
    UI 数据准备节点基类

    提供统一的 UI 数据准备接口，将处理后的数据转换为 UI 可用的格式。

    Subclasses must implement:
        _prepare_display(): 准备 UI 显示数据

    Examples:
        >>> table_node = TableDisplayNode(title="回测结果")
        >>> result = table_node.execute(df)
    """

    _enable_validation: bool = False
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or self.__class__.__name__, config=config, **kwargs)
        self._display_result: Optional[UIDisplayResult] = None

    @abstractmethod
    def _prepare_display(self, input_data: Any, **kwargs) -> UIDisplayResult:
        """
        准备 UI 显示数据

        Args:
            input_data: 输入数据
            **kwargs: 额外参数

        Returns:
            UI 显示结果
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> UIDisplayResult:
        """
        执行 UI 数据准备

        Args:
            input_data: 输入数据
            **kwargs: 额外参数

        Returns:
            UI 显示结果
        """
        self._display_result = self._prepare_display(input_data, **kwargs)
        return self._display_result


class TableDisplayNode(UINode):
    """
    表格显示节点

    将 DataFrame 或类表格数据转换为 Streamlit 表格组件可用的格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        page_size: int = 50,
        **kwargs
    ):
        super().__init__(name=name or "TableDisplay", config=config, **kwargs)
        self.title = title
        self.page_size = page_size

    def _prepare_display(self, input_data: Any, **kwargs) -> UIDisplayResult:
        """准备表格显示数据"""
        if isinstance(input_data, pd.DataFrame):
            data = input_data
            columns = list(input_data.columns)
        elif isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
            columns = list(input_data.keys())
        else:
            data = input_data
            columns = []

        return UIDisplayResult(
            display_type=DisplayType.TABLE,
            title=self.title,
            data=data,
            columns=columns,
            config={
                'page_size': self.page_size,
                'use_container_width': self.config.get('use_container_width', True),
            }
        )


class ChartDisplayNode(UINode):
    """
    图表显示节点

    将数据转换为 Streamlit 图表组件（Line/Bar/Area/Pie）可用的格式。
    """

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
        self.chart_type = chart_type

    def _prepare_display(self, input_data: Any, **kwargs) -> UIDisplayResult:
        """准备图表显示数据"""
        if isinstance(input_data, pd.DataFrame):
            data = input_data
        elif isinstance(input_data, dict):
            data = pd.DataFrame([input_data])
        else:
            data = input_data

        return UIDisplayResult(
            display_type=DisplayType.CHART,
            title=self.title,
            data=data,
            config={
                'chart_type': self.chart_type,
                'use_container_width': self.config.get('use_container_width', True),
            }
        )


class MetricDisplayNode(UINode):
    """
    指标显示节点

    将单个或多个指标值转换为 Streamlit 指标组件可用的格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        delta: float = None,
        delta_color: str = "off",
        **kwargs
    ):
        super().__init__(name=name or "MetricDisplay", config=config, **kwargs)
        self.title = title
        self.delta = delta
        self.delta_color = delta_color

    def _prepare_display(self, input_data: Any, **kwargs) -> UIDisplayResult:
        """准备指标显示数据"""
        value = input_data
        if isinstance(input_data, dict):
            value = input_data.get('value', input_data)
            if 'delta' in input_data:
                self.delta = input_data['delta']
            if 'delta_color' in input_data:
                self.delta_color = input_data['delta_color']

        return UIDisplayResult(
            display_type=DisplayType.METRIC,
            title=self.title,
            data=value,
            config={
                'delta': self.delta,
                'delta_color': self.delta_color,
            }
        )


class TextDisplayNode(UINode):
    """
    文本显示节点

    将文本内容转换为 Streamlit 文本组件可用的格式。
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        title: str = "",
        markdown: bool = True,
        **kwargs
    ):
        super().__init__(name=name or "TextDisplay", config=config, **kwargs)
        self.title = title
        self.markdown = markdown

    def _prepare_display(self, input_data: Any, **kwargs) -> UIDisplayResult:
        """准备文本显示数据"""
        return UIDisplayResult(
            display_type=DisplayType.TEXT,
            title=self.title,
            data=input_data,
            config={
                'markdown': self.markdown,
            }
        )
