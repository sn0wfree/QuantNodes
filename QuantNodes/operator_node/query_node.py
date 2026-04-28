# coding=utf-8
"""
TableQueryNode - 表查询执行节点

执行 SQL 查询并返回 DataFrame 结果。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from QuantNodes.core.node import BaseNode
from QuantNodes.operator_node.base import OperatorNode
from QuantNodes.operator_node.sql_builder import SQLBuilderNode


class TableQueryNode(OperatorNode):
    """
    表查询执行节点

    执行 SQL 查询或使用 SQLBuilderNode 构建查询，返回 DataFrame。

    Examples:
        >>> # 直接执行 SQL
        >>> node = TableQueryNode(db_node=clickhouse_node, sql="SELECT * FROM users")
        >>> df = node.execute()
        >>>
        >>> # 使用链式调用
        >>> node = TableQueryNode(db_node=clickhouse_node)
        >>> df = (node
        ...     .from_table("users")
        ...     .select(["id", "name"])
        ...     .where(["active = 1"])
        ...     .execute())
    """

    def __init__(
        self,
        db_node: Optional[BaseNode] = None,
        sql: Optional[str] = None,
        builder: Optional[SQLBuilderNode] = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            db_node: 数据库节点 (DatabaseNode 实例)
            sql: 直接执行的 SQL 语句
            builder: SQLBuilderNode 用于构建查询
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(name=name or "TableQuery", config=config, **kwargs)
        self._db_node = db_node
        self._sql = sql
        self._builder = builder

        self._table: Optional[str] = None
        self._columns: Optional[List[str]] = None
        self._where: List[str] = []
        self._group_by: List[str] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None

    def from_table(self, table: str) -> 'TableQueryNode':
        """设置查询表"""
        self._table = table
        return self

    def select(self, columns: List[str]) -> 'TableQueryNode':
        """选择列"""
        self._columns = columns
        return self

    def where(self, conditions: List[str]) -> 'TableQueryNode':
        """添加 WHERE 条件"""
        self._where.extend(conditions)
        return self

    def group_by(self, columns: List[str]) -> 'TableQueryNode':
        """添加 GROUP BY"""
        self._group_by.extend(columns)
        return self

    def order_by(self, columns: List[str]) -> 'TableQueryNode':
        """添加 ORDER BY"""
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> 'TableQueryNode':
        """添加 LIMIT"""
        self._limit = n
        return self

    def _execute_operation(self, input_data: Any = None, **kwargs) -> Any:
        """执行查询"""
        if self._db_node is None:
            raise ValueError("db_node is required for TableQueryNode")

        if self._sql:
            sql = self._sql
        elif self._builder:
            sql = self._builder.to_sql()
        else:
            sql = self._build_sql()

        return self._db_node.execute(sql, **kwargs)

    def _build_sql(self) -> str:
        """构建 SQL 语句"""
        if self._table is None:
            raise ValueError("table or sql must be set")

        builder = SQLBuilderNode(table=self._table, columns=self._columns or ['*'])
        
        if self._where:
            builder.where(self._where)
        if self._group_by:
            builder.group_by(self._group_by)
        if self._order_by:
            builder.order_by(self._order_by)
        if self._limit:
            builder.limit(self._limit)

        return builder.to_sql()

    def __repr__(self) -> str:
        return f"<TableQueryNode table='{self._table}'>"
