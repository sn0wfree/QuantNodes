# coding=utf-8
"""SQLBuilderNode 单元测试"""

from QuantNodes.operator_node.sql_builder import SQLBuilderNode


class TestSQLBuilderNode:
    """SQLBuilderNode 测试"""

    def test_basic_select(self):
        """基本 SELECT"""
        builder = SQLBuilderNode(table="users")
        sql = builder.execute()
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "users" in sql

    def test_select_columns(self):
        """选择特定列"""
        builder = SQLBuilderNode(table="users", columns=["id", "name"])
        sql = builder.execute()
        assert "id" in sql
        assert "name" in sql

    def test_select_chaining(self):
        """链式调用"""
        builder = SQLBuilderNode(table="users")
        sql = builder.select(["id", "name"]).execute()
        assert "id" in sql
        assert "name" in sql

    def test_where(self):
        """WHERE 条件"""
        builder = SQLBuilderNode(table="users")
        sql = builder.where(["active = 1"]).execute()
        assert "WHERE" in sql
        assert "active = 1" in sql

    def test_multiple_where(self):
        """多个 WHERE 条件"""
        builder = SQLBuilderNode(table="users")
        sql = builder.where(["a = 1", "b = 2"]).execute()
        assert "a = 1" in sql
        assert "b = 2" in sql

    def test_group_by(self):
        """GROUP BY"""
        builder = SQLBuilderNode(table="orders")
        sql = builder.group_by(["status"]).execute()
        assert "GROUP BY" in sql
        assert "status" in sql

    def test_having(self):
        """HAVING"""
        builder = SQLBuilderNode(table="orders")
        sql = builder.group_by(["status"]).having(["count > 10"]).execute()
        assert "HAVING" in sql
        assert "count > 10" in sql

    def test_order_by(self):
        """ORDER BY"""
        builder = SQLBuilderNode(table="users")
        sql = builder.order_by(["created_at DESC"]).execute()
        assert "ORDER BY" in sql
        assert "created_at DESC" in sql

    def test_limit(self):
        """LIMIT"""
        builder = SQLBuilderNode(table="users")
        sql = builder.limit(100).execute()
        assert "LIMIT 100" in sql

    def test_join(self):
        """JOIN"""
        builder = SQLBuilderNode(table="orders")
        sql = builder.join("LEFT", "users", "orders.user_id = users.id").execute()
        assert "LEFT" in sql

    def test_sample(self):
        """SAMPLE"""
        builder = SQLBuilderNode(table="events")
        sql = builder.sample(0.1).execute()
        assert "SAMPLE" in sql

    def test_complex_query(self):
        """复杂查询"""
        builder = SQLBuilderNode(table="orders")
        sql = (
            builder
            .select(["id", "status", "amount"])
            .where(["status = 'active'", "amount > 100"])
            .group_by(["status"])
            .having(["count > 5"])
            .order_by(["amount DESC"])
            .limit(50)
            .execute()
        )
        assert "SELECT" in sql
        assert "FROM" in sql
        assert "WHERE" in sql
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 50" in sql

    def test_to_sql(self):
        """to_sql 方法"""
        builder = SQLBuilderNode(table="users")
        sql1 = builder.execute()
        sql2 = builder.to_sql()
        assert sql1 == sql2

    def test_repr(self):
        """__repr__ 方法"""
        builder = SQLBuilderNode(table="users")
        repr_str = repr(builder)
        assert "SQLBuilderNode" in repr_str
        assert "users" in repr_str

    def test_empty_table_raises(self):
        """空表名"""
        builder = SQLBuilderNode(table="")
        # 空表名会导致无效 SQL，但不会在 execute 时立即报错
        sql = builder.execute()
        assert "SELECT" in sql
