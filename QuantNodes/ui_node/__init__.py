# coding=utf-8
"""
DisplayNode - 数据可视化节点模块

提供通用数据可视化节点，用于 Pipeline 中数据预处理。
"""
from QuantNodes.ui_node.base import (
    VisualizationType,
    VisualizationData,
    DisplayNode,
    TableDisplayNode,
    ChartDisplayNode,
    MetricDisplayNode,
    TextDisplayNode,
)

__all__ = [
    "VisualizationType",
    "VisualizationData",
    "DisplayNode",
    "TableDisplayNode",
    "ChartDisplayNode",
    "MetricDisplayNode",
    "TextDisplayNode",
]

# 向后兼容别名
DisplayType = VisualizationType
UIDisplayResult = VisualizationData
UINode = DisplayNode