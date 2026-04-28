# coding=utf-8
"""
符号计算引擎 - SQL 编译器

使用 Visitor 模式将 SQLExpression AST 编译为 SQL 字符串。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from QuantNodes.symbolic.dialect import SQLDialect
    from QuantNodes.symbolic.expression import SQLExpression


class SQLCompiler:
    """
    SQL 编译器

    将 SQLExpression AST 编译为 SQL 字符串。

    Examples:
        >>> from QuantNodes.symbolic import ClickHouseDialect, ColumnRef, SQLBinaryOp
        >>> compiler = SQLCompiler(ClickHouseDialect())
        >>> expr = ColumnRef("close") + ColumnRef("open")
        >>> sql = compiler.compile(expr)
        >>> print(sql)
        (`close` + `open`)
    """

    def __init__(self, dialect: "SQLDialect"):
        self.dialect = dialect
        self._context: Dict[str, Any] = {}

    def compile(self, expr: "SQLExpression") -> str:
        """编译表达式为 SQL 字符串"""
        return expr.to_sql(self.dialect)

    def compile_to_select(
        self,
        columns: List["SQLExpression"],
        table: str,
        where: Optional["SQLExpression"] = None,
        group_by: Optional[List["SQLExpression"]] = None,
        having: Optional["SQLExpression"] = None,
        order_by: Optional[List[Tuple["SQLExpression", str]]] = None,
        limit: Optional[int] = None,
    ) -> str:
        """编译为完整 SELECT 语句"""
        col_sqls = [col.to_sql(self.dialect) for col in columns]
        cols_str = ", ".join(col_sqls)

        table_sql = self.dialect.quote_identifier(table)

        sql_parts = [f"SELECT {cols_str}", f"FROM {table_sql}"]

        if where is not None:
            sql_parts.append(f"WHERE {where.to_sql(self.dialect)}")

        if group_by is not None:
            group_sqls = [g.to_sql(self.dialect) for g in group_by]
            sql_parts.append(f"GROUP BY {', '.join(group_sqls)}")

        if having is not None:
            sql_parts.append(f"HAVING {having.to_sql(self.dialect)}")

        if order_by is not None:
            order_sqls = []
            for expr, direction in order_by:
                order_sqls.append(f"{expr.to_sql(self.dialect)} {direction}")
            sql_parts.append(f"ORDER BY {', '.join(order_sqls)}")

        if limit is not None:
            sql_parts.append(f"LIMIT {limit}")

        return " ".join(sql_parts)


def compile_expression(
    expr: "SQLExpression",
    dialect: Optional["SQLDialect"] = None,
    dialect_type: Optional[str] = None,
) -> str:
    """
    便捷函数：编译表达式为 SQL 字符串

    Args:
        expr: SQL 表达式
        dialect: SQL 方言实例
        dialect_type: 方言类型名称 (clickhouse/duckdb/mysql)

    Returns:
        SQL 字符串
    """
    if dialect is None and dialect_type is None:
        from QuantNodes.symbolic.dialect import ClickHouseDialect
        dialect = ClickHouseDialect()
    elif dialect is None and dialect_type is not None:
        from QuantNodes.symbolic.dialect import DialectType
        dialect_map = {
            "clickhouse": "ClickHouseDialect",
            "duckdb": "DuckDBDialect",
            "mysql": "MySQLDialect",
        }
        dialect_name = dialect_map.get(dialect_type.lower())
        if dialect_name is None:
            raise ValueError(f"Unknown dialect type: {dialect_type}")
        from QuantNodes.symbolic.dialect import ClickHouseDialect, DuckDBDialect, MySQLDialect
        dialect_cls = locals()[dialect_name]
        dialect = dialect_cls()

    compiler = SQLCompiler(dialect)
    return compiler.compile(expr)
