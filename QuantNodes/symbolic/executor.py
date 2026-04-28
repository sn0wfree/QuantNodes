# coding=utf-8
"""
符号计算引擎 - 执行引擎

在数据库上执行编译后的 SQL 表达式。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd
    from QuantNodes.symbolic.compiler import SQLCompiler
    from QuantNodes.symbolic.expression import SQLExpression


class SQLExecutor:
    """
    SQL 执行引擎

    在数据库连接上执行编译后的 SQL。

    Examples:
        >>> from QuantNodes.symbolic import ClickHouseDialect, SQLExecutor
        >>> from QuantNodes.symbolic.expression import ColumnRef
        >>> executor = SQLExecutor(db_connection, ClickHouseDialect())
        >>> result = executor.execute("SELECT * FROM t")
    """

    def __init__(
        self,
        connection: Any,
        compiler: Optional["SQLCompiler"] = None,
        dialect: Optional["SQLDialect"] = None,
    ):
        """
        Args:
            connection: 数据库连接对象
            compiler: SQL 编译器实例
            dialect: SQL 方言 (如果 compiler 为 None)
        """
        self.connection = connection
        if compiler is not None:
            self.compiler = compiler
        elif dialect is not None:
            from QuantNodes.symbolic.compiler import SQLCompiler
            self.compiler = SQLCompiler(dialect)
        else:
            from QuantNodes.symbolic.dialect import ClickHouseDialect
            from QuantNodes.symbolic.compiler import SQLCompiler
            self.compiler = SQLCompiler(ClickHouseDialect())

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "pd.DataFrame":
        """
        执行 SQL 查询

        Args:
            sql: SQL 字符串
            params: 查询参数

        Returns:
            DataFrame 结果
        """
        import pandas as pd

        cursor = self.connection.cursor()

        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)

        return pd.DataFrame()

    def execute_expression(
        self,
        expr: "SQLExpression",
        table: str,
        columns: Optional[List["SQLExpression"]] = None,
        where: Optional["SQLExpression"] = None,
        group_by: Optional[List["SQLExpression"]] = None,
        having: Optional["SQLExpression"] = None,
        order_by: Optional[List] = None,
        limit: Optional[int] = None,
    ) -> "pd.DataFrame":
        """
        执行表达式

        Args:
            expr: SQL 表达式
            table: 表名
            columns: 选择列
            where: WHERE 条件
            group_by: GROUP BY 列
            having: HAVING 条件
            order_by: ORDER BY 列
            limit: LIMIT

        Returns:
            DataFrame 结果
        """
        if columns is None:
            columns = [expr]
        elif expr not in columns:
            columns = columns + [expr]

        sql = self.compiler.compile_to_select(
            columns=columns,
            table=table,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
        )

        return self.execute(sql)


def execute_sql(
    sql: str,
    connection: Any,
    params: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """
    便捷函数：执行 SQL

    Args:
        sql: SQL 字符串
        connection: 数据库连接
        params: 查询参数

    Returns:
        DataFrame 结果
    """
    executor = SQLExecutor(connection)
    return executor.execute(sql, params)
