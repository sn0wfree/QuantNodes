# coding=utf-8
"""SQLExecutor 单元测试"""
import pandas as pd

from QuantNodes.symbolic.executor import SQLExecutor, execute_sql
from QuantNodes.symbolic.compiler import SQLCompiler
from QuantNodes.symbolic.dialect import ClickHouseDialect
from QuantNodes.symbolic.expression import ColumnRef, LiteralValue


class MockConnection:
    """模拟数据库连接"""

    def cursor(self):
        return MockCursor()

    def commit(self):
        pass

    def close(self):
        pass


class MockCursor:
    """模拟数据库游标"""

    def __init__(self):
        self._data = [
            (1, "Alice"),
            (2, "Bob"),
        ]
        self.description = [
            ("id",),
            ("name",),
        ]

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._data

    def fetchone(self):
        if self._data:
            return self._data.pop(0)
        return None


class MockCursorNoResult:
    """模拟无返回结果的游标"""

    def __init__(self):
        self.description = None

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class TestSQLExecutor:
    """SQLExecutor 测试"""

    def test_executor_init_with_connection(self):
        """使用连接初始化"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        assert executor.connection is conn

    def test_executor_init_with_dialect(self):
        """使用方言初始化"""
        executor = SQLExecutor(connection=None, dialect=ClickHouseDialect())
        assert executor.compiler is not None
        assert isinstance(executor.compiler, SQLCompiler)

    def test_executor_init_with_compiler(self):
        """使用编译器初始化"""
        compiler = SQLCompiler(ClickHouseDialect())
        executor = SQLExecutor(connection=None, compiler=compiler)
        assert executor.compiler is compiler

    def test_executor_default_dialect(self):
        """默认使用 ClickHouseDialect"""
        executor = SQLExecutor(connection=None)
        assert isinstance(executor.compiler.dialect, ClickHouseDialect)

    def test_execute_simple_query(self):
        """执行简单查询"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        result = executor.execute("SELECT * FROM test")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name"]

    def test_execute_with_params(self):
        """执行带参数的查询"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        result = executor.execute("SELECT * FROM test WHERE id = ?", params=(1,))
        assert isinstance(result, pd.DataFrame)

    def test_execute_no_result(self):
        """执行无返回结果的 SQL"""
        conn = MockConnection()
        cursor = MockCursorNoResult()
        conn.cursor = lambda: cursor
        executor = SQLExecutor(connection=conn)
        result = executor.execute("INSERT INTO test VALUES (1, 'Alice')")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_execute_expression(self):
        """执行表达式"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        expr = ColumnRef("close") + ColumnRef("open")
        result = executor.execute_expression(
            expr=expr,
            table="stock",
            columns=[ColumnRef("close"), ColumnRef("open")],
        )
        assert isinstance(result, pd.DataFrame)

    def test_execute_expression_with_where(self):
        """执行带 WHERE 的表达式"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        expr = ColumnRef("close")
        result = executor.execute_expression(
            expr=expr,
            table="stock",
            columns=[ColumnRef("close")],
            where=ColumnRef("close") > LiteralValue(100),
        )
        assert isinstance(result, pd.DataFrame)

    def test_execute_expression_with_limit(self):
        """执行带 LIMIT 的表达式"""
        conn = MockConnection()
        executor = SQLExecutor(connection=conn)
        expr = ColumnRef("close")
        result = executor.execute_expression(
            expr=expr,
            table="stock",
            columns=[ColumnRef("close")],
            limit=10,
        )
        assert isinstance(result, pd.DataFrame)


class TestExecuteSQL:
    """execute_sql 便捷函数测试"""

    def test_execute_sql_basic(self):
        """基本用法"""
        conn = MockConnection()
        result = execute_sql("SELECT * FROM test", connection=conn)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_execute_sql_with_params(self):
        """带参数"""
        conn = MockConnection()
        result = execute_sql("SELECT * FROM test WHERE id = ?", connection=conn, params=(1,))
        assert isinstance(result, pd.DataFrame)
