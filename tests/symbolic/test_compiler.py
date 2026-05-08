# coding=utf-8
"""SQLCompiler 单元测试"""
import pytest

from QuantNodes.symbolic.compiler import SQLCompiler, compile_expression
from QuantNodes.symbolic.expression import (
    ColumnRef, LiteralValue,
)
from QuantNodes.symbolic.dialect import ClickHouseDialect, DuckDBDialect


class TestSQLCompiler:
    """SQLCompiler 测试"""

    def test_compile_basic(self):
        """编译基本表达式"""
        compiler = SQLCompiler(ClickHouseDialect())
        expr = ColumnRef("close")
        sql = compiler.compile(expr)
        assert "`close`" in sql

    def test_compile_binary_op(self):
        """编译二元运算"""
        compiler = SQLCompiler(ClickHouseDialect())
        expr = ColumnRef("close") + ColumnRef("open")
        sql = compiler.compile(expr)
        assert "(" in sql
        assert "`close`" in sql
        assert "`open`" in sql

    def test_compile_to_select_basic(self):
        """编译基本 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
        )
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "`stock`" in sql

    def test_compile_to_select_multiple_columns(self):
        """编译多列 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close"), ColumnRef("open"), ColumnRef("volume")],
            table="stock",
        )
        assert "`close`" in sql
        assert "`open`" in sql
        assert "`volume`" in sql

    def test_compile_to_select_with_where(self):
        """编译带 WHERE 的 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
            where=ColumnRef("close") > LiteralValue(100),
        )
        assert "WHERE" in sql
        assert "> 100" in sql

    def test_compile_to_select_with_group_by(self):
        """编译带 GROUP BY 的 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("code"), ColumnRef("close")],
            table="stock",
            group_by=[ColumnRef("code")],
        )
        assert "GROUP BY" in sql
        assert "`code`" in sql

    def test_compile_to_select_with_having(self):
        """编译带 HAVING 的 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("code")],
            table="stock",
            group_by=[ColumnRef("code")],
            having=ColumnRef("close") > LiteralValue(100),
        )
        assert "HAVING" in sql

    def test_compile_to_select_with_order_by(self):
        """编译带 ORDER BY 的 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
            order_by=[(ColumnRef("close"), "DESC")],
        )
        assert "ORDER BY" in sql
        assert "DESC" in sql

    def test_compile_to_select_with_limit(self):
        """编译带 LIMIT 的 SELECT"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
            limit=100,
        )
        assert "LIMIT 100" in sql

    def test_compile_to_select_full(self):
        """编译完整 SELECT 语句"""
        compiler = SQLCompiler(ClickHouseDialect())
        sql = compiler.compile_to_select(
            columns=[ColumnRef("code"), ColumnRef("close")],
            table="stock",
            where=ColumnRef("close") > LiteralValue(100),
            group_by=[ColumnRef("code")],
            having=ColumnRef("close") > LiteralValue(200),
            order_by=[(ColumnRef("close"), "DESC")],
            limit=50,
        )
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "WHERE" in sql
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 50" in sql


class TestCompileExpression:
    """compile_expression 便捷函数测试"""

    def test_compile_expression_basic(self):
        """基本用法"""
        expr = ColumnRef("close") + ColumnRef("open")
        sql = compile_expression(expr)
        assert "`close`" in sql
        assert "`open`" in sql

    def test_compile_expression_with_dialect(self):
        """指定方言"""
        expr = ColumnRef("close")
        sql_ch = compile_expression(expr, dialect=ClickHouseDialect())
        sql_duckdb = compile_expression(expr, dialect=DuckDBDialect())
        assert "`close`" in sql_ch
        assert '"close"' in sql_duckdb

    def test_compile_expression_with_dialect_type(self):
        """指定方言类型"""
        expr = ColumnRef("close")
        sql_ch = compile_expression(expr, dialect_type="clickhouse")
        sql_duckdb = compile_expression(expr, dialect_type="duckdb")
        sql_mysql = compile_expression(expr, dialect_type="mysql")
        assert "`close`" in sql_ch
        assert '"close"' in sql_duckdb
        assert "`close`" in sql_mysql

    def test_compile_expression_invalid_dialect(self):
        """无效方言类型"""
        expr = ColumnRef("close")
        with pytest.raises(ValueError):
            compile_expression(expr, dialect_type="invalid")

    def test_compile_expression_default_dialect(self):
        """默认方言为 ClickHouse"""
        expr = ColumnRef("close")
        sql = compile_expression(expr)
        assert "`close`" in sql
