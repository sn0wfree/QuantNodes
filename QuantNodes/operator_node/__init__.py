# coding=utf-8
"""
OperatorNode 操作节点模块

提供数据操作、SQL 构建和转换的节点类型：

- OperatorNode: 操作节点基类
- ChainOperator: 链式操作节点
- SQLBuilderNode: SQL 构建节点
- TableQueryNode: 表查询执行节点
- TransformNode: 数据转换节点

Examples:
    >>> # SQL 构建
    >>> builder = SQLBuilderNode(table="users")
    >>> sql = builder.select(["id", "name"]).where(["active = 1"]).to_sql()
    
    >>> # 数据转换
    >>> transformer = TransformNode()
    >>> result = transformer.select(["col1", "col2"]).filter(lambda df: df["value"] > 0).execute(df)
    
    >>> # 链式操作
    >>> pipeline = query_node >> TransformNode().select(["name"])
"""

from QuantNodes.operator_node.base import OperatorNode, ChainOperator
from QuantNodes.operator_node.sql_builder import SQLBuilderNode
from QuantNodes.operator_node.query_node import TableQueryNode
from QuantNodes.operator_node.transform import TransformNode
from QuantNodes.operator_node.SQLUtils import SQLBuilder
from QuantNodes.operator_node.TableNode import OperatorBaseNode, Node2
from QuantNodes.operator_node import TableOperator

__all__ = [
    'OperatorNode',
    'ChainOperator',
    'SQLBuilderNode',
    'TableQueryNode',
    'TransformNode',
    'SQLBuilder',
    'OperatorBaseNode',
    'Node2',
    'TableOperator',
]
