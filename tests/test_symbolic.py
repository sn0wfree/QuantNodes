# -*- coding: utf-8 -*-
"""Symbolic computation engine unit tests"""

import sys
import pytest

sys.path.insert(0, '/home/ll/Public/QuantNodes/QuantNodes')

from symbolic import (
    SQLCompiler,
    ColumnRef,
    LiteralValue,
    SQLBinaryOp,
    SQLUnaryOp,
    SQLComparison,
    SQLLogicalOp,
    SQLFunction,
    SQLCase,
    ClickHouseDialect,
    DuckDBDialect,
    MySQLDialect,
    TechnicalFunctions,
    TA_FUNCTIONS,
)
from symbolic.optimizer import SQLOptimizer, optimize_expression


class TestColumnRef:
    """Tests for ColumnRef"""

    def test_column_ref_ch(self):
        """Test ClickHouse column reference"""
        dialect = ClickHouseDialect()
        col = ColumnRef("close")
        sql = col.to_sql(dialect)
        assert sql == "`close`"

    def test_column_ref_duckdb(self):
        """Test DuckDB column reference"""
        dialect = DuckDBDialect()
        col = ColumnRef("close")
        sql = col.to_sql(dialect)
        assert sql == '"close"'

    def test_column_ref_mysql(self):
        """Test MySQL column reference"""
        dialect = MySQLDialect()
        col = ColumnRef("close")
        sql = col.to_sql(dialect)
        assert sql == "`close`"

    def test_column_ref_with_table(self):
        """Test column reference with table prefix"""
        dialect = ClickHouseDialect()
        col = ColumnRef("close", "t1")
        sql = col.to_sql(dialect)
        assert sql == "`t1`.`close`"


class TestLiteralValue:
    """Tests for LiteralValue"""

    def test_literal_int(self):
        """Test integer literal"""
        dialect = ClickHouseDialect()
        lit = LiteralValue(100)
        sql = lit.to_sql(dialect)
        assert sql == "100"

    def test_literal_float(self):
        """Test float literal"""
        dialect = ClickHouseDialect()
        lit = LiteralValue(3.14)
        sql = lit.to_sql(dialect)
        assert sql == "3.14"

    def test_literal_string(self):
        """Test string literal"""
        dialect = ClickHouseDialect()
        lit = LiteralValue("test")
        sql = lit.to_sql(dialect)
        assert sql == "'test'"

    def test_literal_none(self):
        """Test None literal"""
        dialect = ClickHouseDialect()
        lit = LiteralValue(None)
        sql = lit.to_sql(dialect)
        assert sql == "NULL"

    def test_literal_bool(self):
        """Test boolean literal"""
        dialect = ClickHouseDialect()
        lit_true = LiteralValue(True)
        lit_false = LiteralValue(False)
        assert lit_true.to_sql(dialect) == "1"
        assert lit_false.to_sql(dialect) == "0"


class TestSQLBinaryOp:
    """Tests for SQLBinaryOp"""

    def test_binary_add(self):
        """Test addition"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") + ColumnRef("open")
        sql = expr.to_sql(dialect)
        assert sql == "(`close` + `open`)"

    def test_binary_sub(self):
        """Test subtraction"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") - ColumnRef("open")
        sql = expr.to_sql(dialect)
        assert sql == "(`close` - `open`)"

    def test_binary_mul(self):
        """Test multiplication"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") * LiteralValue(2)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` * 2)"

    def test_binary_div(self):
        """Test division"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") / ColumnRef("open")
        sql = expr.to_sql(dialect)
        assert sql == "(`close` / `open`)"


class TestSQLComparison:
    """Tests for SQLComparison"""

    def test_comparison_gt(self):
        """Test greater than"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") > LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` > 100)"

    def test_comparison_ge(self):
        """Test greater than or equal"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") >= LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` >= 100)"

    def test_comparison_lt(self):
        """Test less than"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") < LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` < 100)"

    def test_comparison_le(self):
        """Test less than or equal"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") <= LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` <= 100)"

    def test_comparison_eq(self):
        """Test equality"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") == LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` == 100)"

    def test_comparison_ne(self):
        """Test not equal"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") != LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert sql == "(`close` != 100)"


class TestSQLLogicalOp:
    """Tests for SQLLogicalOp"""

    def test_logical_and(self):
        """Test AND"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") > LiteralValue(100) & ColumnRef("open") < LiteralValue(200)
        sql = expr.to_sql(dialect)
        assert "AND" in sql

    def test_logical_or(self):
        """Test OR"""
        dialect = ClickHouseDialect()
        expr = ColumnRef("close") > LiteralValue(100) | ColumnRef("open") < LiteralValue(50)
        sql = expr.to_sql(dialect)
        assert "OR" in sql


class TestSQLCompiler:
    """Tests for SQLCompiler"""

    def test_compile_simple_expr(self):
        """Test simple expression compilation"""
        dialect = ClickHouseDialect()
        compiler = SQLCompiler(dialect)
        expr = ColumnRef("close") + ColumnRef("open")
        sql = compiler.compile(expr)
        assert "`close`" in sql and "`open`" in sql

    def test_compile_select(self):
        """Test SELECT statement compilation"""
        dialect = ClickHouseDialect()
        compiler = SQLCompiler(dialect)
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close"), ColumnRef("open")],
            table="stock",
        )
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "`close`" in sql
        assert "`stock`" in sql

    def test_compile_select_with_where(self):
        """Test SELECT with WHERE clause"""
        dialect = ClickHouseDialect()
        compiler = SQLCompiler(dialect)
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
            where=ColumnRef("close") > LiteralValue(100),
        )
        assert "WHERE" in sql
        assert "> 100" in sql

    def test_compile_select_with_limit(self):
        """Test SELECT with LIMIT"""
        dialect = ClickHouseDialect()
        compiler = SQLCompiler(dialect)
        sql = compiler.compile_to_select(
            columns=[ColumnRef("close")],
            table="stock",
            limit=10,
        )
        assert "LIMIT 10" in sql


class TestDialectDifferences:
    """Tests for dialect-specific SQL generation"""

    def test_identifier_quoting(self):
        """Test identifier quoting differences"""
        ch = ClickHouseDialect()
        duckdb = DuckDBDialect()
        mysql = MySQLDialect()

        col = ColumnRef("test")

        assert col.to_sql(ch) == "`test`"
        assert col.to_sql(duckdb) == '"test"'
        assert col.to_sql(mysql) == "`test`"

    def test_function_names(self):
        """Test function name differences"""
        ch = ClickHouseDialect()
        mysql = MySQLDialect()

        assert ch.func_stddev("x") == "stddevPop(x)"
        assert mysql.func_stddev("x") == "STDDEV_POP(x)"

        assert ch.func_ceil("x") == "ceil(x)"
        assert mysql.func_ceil("x") == "CEILING(x)"


class TestTechnicalFunctions:
    """Tests for TechnicalFunctions"""

    def test_sma(self):
        """Test SMA function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.sma(expr, 20)
        assert isinstance(result, SQLFunction)

    def test_ema(self):
        """Test EMA function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.ema(expr, 12)
        assert isinstance(result, SQLFunction)

    def test_delay(self):
        """Test delay function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.delay(expr, 1)
        assert isinstance(result, SQLFunction)

    def test_delta(self):
        """Test delta function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.delta(expr, 1)
        assert isinstance(result, SQLBinaryOp)

    def test_pct_change(self):
        """Test pct_change function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.pct_change(expr, 1)
        assert isinstance(result, SQLBinaryOp)

    def test_rank(self):
        """Test rank function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.rank(expr)
        assert isinstance(result, SQLFunction)

    def test_abs(self):
        """Test abs function"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.abs(expr)
        assert isinstance(result, SQLFunction)

    def test_ta_functions_dict(self):
        """Test TA_FUNCTIONS dictionary"""
        assert "sma" in TA_FUNCTIONS
        assert "ema" in TA_FUNCTIONS
        assert "delay" in TA_FUNCTIONS
        assert "delta" in TA_FUNCTIONS
        assert "rank" in TA_FUNCTIONS


class TestSQLOptimizer:
    """Tests for SQLOptimizer"""

    def test_optimize_column_ref(self):
        """Test that column references pass through optimizer"""
        col = ColumnRef("close")
        optimizer = SQLOptimizer()
        result = optimizer.optimize(col)
        assert isinstance(result, ColumnRef)

    def test_optimize_literal(self):
        """Test that literals pass through optimizer"""
        lit = LiteralValue(100)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(lit)
        assert isinstance(result, LiteralValue)


class TestSQLExpressionOperators:
    """Tests for SQLExpression operators"""

    def test_chaining_operations(self):
        """Test chaining multiple operations"""
        dialect = ClickHouseDialect()
        expr = (ColumnRef("close") - ColumnRef("open")) / ColumnRef("open") * LiteralValue(100)
        sql = expr.to_sql(dialect)
        assert "`close`" in sql
        assert "`open`" in sql
        assert "* 100" in sql

    def test_complex_expression(self):
        """Test complex expression"""
        dialect = ClickHouseDialect()
        expr = (ColumnRef("close") > LiteralValue(100)) & (ColumnRef("volume") > LiteralValue(1000))
        sql = expr.to_sql(dialect)
        assert "`close`" in sql
        assert "`volume`" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
