# coding=utf-8
"""TableQueryNode 单元测试"""
import pytest
from unittest.mock import MagicMock, Mock

from QuantNodes.operator_node.query_node import TableQueryNode
from QuantNodes.operator_node.sql_builder import SQLBuilderNode


class MockDBNode:
    """模拟数据库节点"""

    def __init__(self):
        self.executed_sql = None
        self._data = {"result": "mock_data"}

    def execute(self, sql, **kwargs):
        self.executed_sql = sql
        return self._data


class TestTableQueryNode:
    """TableQueryNode 测试"""

    def test_creation(self):
        """创建节点"""
        node = TableQueryNode(name="TestQuery")
        assert node.name == "TestQuery"

    def test_from_table(self):
        """设置表名"""
        node = TableQueryNode()
        result = node.from_table("users")
        assert result is node
        assert node._table == "users"

    def test_select_columns(self):
        """选择列"""
        node = TableQueryNode()
        result = node.select(["id", "name"])
        assert result is node
        assert node._columns == ["id", "name"]

    def test_where_conditions(self):
        """WHERE 条件"""
        node = TableQueryNode()
        result = node.where(["active = 1"])
        assert result is node
        assert "active = 1" in node._where

    def test_group_by_columns(self):
        """GROUP BY 列"""
        node = TableQueryNode()
        result = node.group_by(["status"])
        assert result is node
        assert "status" in node._group_by

    def test_order_by_columns(self):
        """ORDER BY 列"""
        node = TableQueryNode()
        result = node.order_by(["created_at DESC"])
        assert result is node
        assert "created_at DESC" in node._order_by

    def test_limit(self):
        """LIMIT"""
        node = TableQueryNode()
        result = node.limit(100)
        assert result is node
        assert node._limit == 100

    def test_chaining(self):
        """链式调用"""
        node = TableQueryNode()
        result = (
            node
            .from_table("users")
            .select(["id", "name"])
            .where(["active = 1"])
            .limit(10)
        )
        assert result is node
        assert node._table == "users"
        assert node._columns == ["id", "name"]
        assert node._limit == 10

    def test_execute_with_direct_sql(self):
        """执行直接 SQL"""
        db_node = MockDBNode()
        node = TableQueryNode(db_node=db_node, sql="SELECT * FROM users")
        result = node.execute()
        assert result == {"result": "mock_data"}
        assert db_node.executed_sql == "SELECT * FROM users"

    def test_execute_without_db_node(self):
        """无数据库节点时应报错"""
        from QuantNodes.core.node import NodeExecutionError
        node = TableQueryNode()
        node._sql = "SELECT * FROM users"
        with pytest.raises(NodeExecutionError, match="db_node is required"):
            node.execute()

    def test_execute_builds_sql(self):
        """执行时构建 SQL"""
        db_node = MockDBNode()
        node = TableQueryNode(db_node=db_node)
        node.from_table("users")
        node.select(["id", "name"])
        node.where(["active = 1"])
        node.limit(10)
        node.execute()
        assert "SELECT" in db_node.executed_sql
        assert "FROM" in db_node.executed_sql

    def test_execute_with_builder(self):
        """使用 SQLBuilderNode"""
        db_node = MockDBNode()
        builder = SQLBuilderNode(table="users", columns=["id", "name"])
        node = TableQueryNode(db_node=db_node, builder=builder)
        node.execute()
        assert db_node.executed_sql is not None

    def test_repr(self):
        """__repr__ 方法"""
        node = TableQueryNode()
        node.from_table("users")
        repr_str = repr(node)
        assert "TableQueryNode" in repr_str
        assert "users" in repr_str
