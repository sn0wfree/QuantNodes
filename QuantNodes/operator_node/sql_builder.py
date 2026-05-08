# coding=utf-8
"""
SQLBuilderNode - SQL 构建节点

基于 SQLUtils 构建的 SQL 生成节点。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from QuantNodes.operator_node.base import OperatorNode


class SQLBuilderNode(OperatorNode):
    """
    SQL 构建节点

    构建 SQL 查询语句，支持链式调用。

    Examples:
        >>> builder = SQLBuilderNode(table="users")
        >>> sql = builder.select(["id", "name"]).where(["active = 1"]).execute()
        >>> print(sql)
        SELECT id, name FROM users WHERE active = 1
    """

    def __init__(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            table: 表名 (格式: db.table 或 table)
            columns: 要选择的列，None 表示 *
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(name=name or "SQLBuilder", config=config, **kwargs)
        self._table = table
        self._columns = columns or ['*']
        self._where: List[str] = []
        self._group_by: List[str] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None
        self._having: List[str] = []
        self._joins: List[Dict[str, Any]] = []
        self._sample: Optional[str] = None

    def select(self, columns: List[str]) -> 'SQLBuilderNode':
        """选择列"""
        self._columns = columns
        return self

    def where(self, conditions: List[str]) -> 'SQLBuilderNode':
        """添加 WHERE 条件"""
        self._where.extend(conditions)
        return self

    def group_by(self, columns: List[str]) -> 'SQLBuilderNode':
        """添加 GROUP BY"""
        self._group_by.extend(columns)
        return self

    def having(self, conditions: List[str]) -> 'SQLBuilderNode':
        """添加 HAVING 条件"""
        self._having.extend(conditions)
        return self

    def order_by(self, columns: List[str]) -> 'SQLBuilderNode':
        """添加 ORDER BY"""
        self._order_by.extend(columns)
        return self

    def limit(self, n: int) -> 'SQLBuilderNode':
        """添加 LIMIT"""
        self._limit = n
        return self

    def join(self, join_type: str, table: str, condition: str) -> 'SQLBuilderNode':
        """添加 JOIN"""
        self._joins.append({
            'type': join_type,
            'table': table,
            'condition': condition
        })
        return self

    def sample(self, ratio: str) -> 'SQLBuilderNode':
        """添加 SAMPLE"""
        self._sample = ratio
        return self

    def _execute_operation(self, input_data: Any = None, **kwargs) -> str:
        """生成 SQL 语句"""
        from QuantNodes.operator_node.sql_utils import SQLBuilder

        cols_str = ','.join(self._columns)
        db_table = self._table

        sql = SQLBuilder.create_select_sql(
            DB_TABLE=db_table,
            cols=self._columns if self._columns != ['*'] else ['*'],
            sample=self._sample,
            array_join=None,
            join=self._joins[0] if self._joins else None,
            prewhere=None,
            where=self._where if self._where else None,
            having=self._having if self._having else None,
            group_by=self._group_by if self._group_by else None,
            order_by=self._order_by if self._order_by else None,
            limit_by=None,
            limit=self._limit
        )
        return sql

    def to_sql(self) -> str:
        """返回 SQL 语句（execute 的别名）"""
        return self.execute()

    def __repr__(self) -> str:
        return f"<SQLBuilderNode table='{self._table}'>"
