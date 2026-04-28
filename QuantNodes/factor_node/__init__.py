# coding=utf-8
"""因子节点模块

提供因子数据库、因子表、因子类和因子运算功能。
"""
import numpy as np
import pandas as pd

from QuantNodes.factor_node.factor_db import FactorDB, WritableFactorDB
from QuantNodes.factor_node.factor_table import FactorTable, CustomFT
from QuantNodes.factor_node.factor import Factor, DataFactor, Factorize
from QuantNodes.factor_node.factor_operation import (
    DerivativeFactor,
    PointOperation,
    TimeOperation,
    SectionOperation,
    PanelOperation,
)

from QuantNodes.factor_node.factor_node import FactorNode, FactorPipeline
from QuantNodes.factor_node.point_factor import PointFactorNode, ArithmeticFactorNode
from QuantNodes.factor_node.time_factor import TimeFactorNode, ExpandingFactorNode
from QuantNodes.factor_node.cross_section_factor import CrossSectionFactorNode, GroupRankFactorNode
from QuantNodes.factor_node.panel_factor import PanelFactorNode, DelayFactorNode, DeltaFactorNode

__all__ = [
    # factor_db
    'FactorDB',
    'WritableFactorDB',

    # factor_table
    'FactorTable',
    'CustomFT',

    # factor
    'Factor',
    'DataFactor',
    'Factorize',

    # factor_operation
    'DerivativeFactor',
    'PointOperation',
    'TimeOperation',
    'SectionOperation',
    'PanelOperation',

    # FactorNode (BaseNode integration)
    'FactorNode',
    'FactorPipeline',

    # PointFactorNode
    'PointFactorNode',
    'ArithmeticFactorNode',

    # TimeFactorNode
    'TimeFactorNode',
    'ExpandingFactorNode',

    # CrossSectionFactorNode
    'CrossSectionFactorNode',
    'GroupRankFactorNode',

    # PanelFactorNode
    'PanelFactorNode',
    'DelayFactorNode',
    'DeltaFactorNode',
]

if __name__ == '__main__':
    pass
