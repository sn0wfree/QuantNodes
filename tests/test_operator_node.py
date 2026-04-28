# -*- coding: utf-8 -*-
"""OperatorNode unit tests"""
import pytest
import pandas as pd

from QuantNodes.operator_node import (
    OperatorNode,
    ChainOperator,
    SQLBuilderNode,
    TableQueryNode,
    TransformNode,
    SQLBuilder,
)


class DummyOperator(OperatorNode):
    """测试用操作节点"""
    def __init__(self, return_value="dummy_result", name=None):
        super().__init__(name=name or "Dummy")
        self.return_value = return_value

    def _execute_operation(self, input_data=None, **kwargs):
        return self.return_value


class TestOperatorNode:
    """Tests for OperatorNode base class"""

    def test_operator_node_initialization(self):
        """测试 OperatorNode 初始化"""
        op = DummyOperator()
        assert op.name == "Dummy"
        assert op.state.value == "idle"

    def test_operator_node_execute(self):
        """测试执行"""
        op = DummyOperator(return_value="test_result")
        result = op.execute()
        assert result == "test_result"

    def test_operator_node_chain(self):
        """测试链式调用"""
        op1 = DummyOperator(return_value="first")
        op2 = DummyOperator(return_value="second")
        chain = op1.then(op2)
        assert isinstance(chain, ChainOperator)
        assert len(chain.operators) == 2

    def test_operator_rshift(self):
        """测试 >> 运算符"""
        op1 = DummyOperator()
        op2 = DummyOperator()
        chain = op1 >> op2
        assert isinstance(chain, ChainOperator)


class TestChainOperator:
    """Tests for ChainOperator"""

    def test_chain_execute(self):
        """测试链式执行"""
        op1 = DummyOperator(return_value="first")
        op2 = DummyOperator(return_value="second")
        chain = ChainOperator([op1, op2])
        result = chain.execute()
        assert result == "second"

    def test_chain_with_input(self):
        """测试带输入的链式执行"""
        results = []
        
        class RecordOperator(OperatorNode):
            def _execute_operation(self, input_data=None, **kwargs):
                results.append(input_data)
                return f"processed_{input_data}"

        op1 = RecordOperator()
        op2 = RecordOperator()
        chain = ChainOperator([op1, op2])
        
        result = chain.execute("input")
        assert result == "processed_processed_input"
        assert results == ["input", "processed_input"]


class TestSQLBuilderNode:
    """Tests for SQLBuilderNode"""

    def test_sql_builder_basic(self):
        """测试基本 SQL 构建"""
        builder = SQLBuilderNode(table="users")
        builder.select(["id", "name"]).where(["active = 1"]).limit(10)
        sql = builder.to_sql()
        
        assert "SELECT" in sql
        assert "FROM users" in sql
        assert "WHERE active = 1" in sql
        assert "LIMIT 10" in sql

    def test_sql_builder_where_chain(self):
        """测试 WHERE 链式调用"""
        builder = SQLBuilderNode(table="users")
        builder.where(["a = 1"]).where(["b = 2"])
        sql = builder.to_sql()
        
        assert "a = 1" in sql
        assert "b = 2" in sql

    def test_sql_builder_group_by(self):
        """测试 GROUP BY"""
        builder = SQLBuilderNode(table="orders")
        builder.select(["status", "count(*) as cnt"]).group_by(["status"])
        sql = builder.to_sql()
        
        assert "GROUP BY" in sql
        assert "status" in sql

    def test_sql_builder_order_by(self):
        """测试 ORDER BY"""
        builder = SQLBuilderNode(table="users")
        builder.select(["id", "name"]).order_by(["created_at desc"])
        sql = builder.to_sql()
        
        assert "ORDER BY" in sql
        assert "created_at desc" in sql

    def test_sql_builder_repr(self):
        """测试 __repr__"""
        builder = SQLBuilderNode(table="users")
        assert "users" in repr(builder)


class TestTransformNode:
    """Tests for TransformNode"""

    @pytest.fixture
    def sample_df(self):
        """测试用 DataFrame"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'score': [85.5, 90.0, 78.5, 92.0, 88.0]
        })

    def test_select(self, sample_df):
        """测试选择列"""
        transformer = TransformNode().select(["id", "name"])
        result = transformer.execute(sample_df)
        
        assert list(result.columns) == ["id", "name"]
        assert len(result) == 5

    def test_filter_lambda(self, sample_df):
        """测试过滤行（lambda）"""
        transformer = TransformNode().filter(lambda df: df["score"] >= 90)
        result = transformer.execute(sample_df)
        
        assert len(result) == 2
        assert result.iloc[0]["name"] == "Bob"

    def test_filter_string(self, sample_df):
        """测试过滤行（字符串）"""
        transformer = TransformNode().filter("score >= 90")
        result = transformer.execute(sample_df)
        
        assert len(result) == 2

    def test_aggregate(self, sample_df):
        """测试聚合"""
        df_with_category = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        transformer = TransformNode().aggregate(
            group_by=["category"],
            agg={"value": "sum"}
        )
        result = transformer.execute(df_with_category)
        
        assert len(result) == 2
        assert result[result["category"] == "A"]["value"].iloc[0] == 30
        assert result[result["category"] == "B"]["value"].iloc[0] == 120

    def test_sort_by(self, sample_df):
        """测试排序"""
        transformer = TransformNode().sort_by("score", ascending=False)
        result = transformer.execute(sample_df)
        
        assert result.iloc[0]["name"] == "David"
        assert result.iloc[-1]["name"] == "Charlie"

    def test_rename(self, sample_df):
        """测试重命名列"""
        transformer = TransformNode().rename({"name": "full_name"})
        result = transformer.execute(sample_df)
        
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_fillna(self):
        """测试填充空值"""
        df_with_null = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [None, 'x', 'y', None]
        })
        transformer = TransformNode().fillna(0)
        result = transformer.execute(df_with_null)
        
        assert result['a'].iloc[2] == 0
        assert result['b'].iloc[0] == 0
        assert result['b'].iloc[-1] == 0

    def test_chain_transforms(self, sample_df):
        """测试链式转换"""
        transformer = (
            TransformNode()
            .select(["name", "score"])
            .filter("score >= 85")
            .sort_by("score", ascending=False)
        )
        result = transformer.execute(sample_df)
        
        assert len(result) == 4
        assert result.iloc[0]["name"] == "David"

    def test_transform_repr(self):
        """测试 __repr__"""
        transformer = TransformNode().select(["a", "b"])
        assert "TransformNode" in repr(transformer)
        assert "operations=1" in repr(transformer)


class TestSQLBuilder:
    """Tests for SQLBuilder utility class"""

    def test_create_select_sql_basic(self):
        """测试基本 SELECT SQL"""
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="users",
            cols=["id", "name"],
            sample=None,
            array_join=None,
            join=None,
            prewhere=None,
            where=None,
            having=None,
            group_by=None,
            order_by=None,
            limit_by=None,
            limit=None
        )
        
        assert "SELECT id,name" in sql
        assert "FROM users" in sql

    def test_create_select_sql_with_where(self):
        """测试带 WHERE 的 SELECT"""
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="users",
            cols=["*"],
            sample=None,
            array_join=None,
            join=None,
            prewhere=None,
            where=["active = 1", "age > 18"],
            having=None,
            group_by=None,
            order_by=None,
            limit_by=None,
            limit=None
        )
        
        assert "WHERE" in sql
        assert "active = 1" in sql
        assert "age > 18" in sql

    def test_create_select_sql_with_group_by(self):
        """测试带 GROUP BY 的 SELECT"""
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="orders",
            cols=["status", "count(*)"],
            sample=None,
            array_join=None,
            join=None,
            prewhere=None,
            where=None,
            having=None,
            group_by=["status"],
            order_by=None,
            limit_by=None,
            limit=None
        )
        
        assert "GROUP BY (status)" in sql

    def test_create_select_sql_with_limit(self):
        """测试带 LIMIT 的 SELECT"""
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="users",
            cols=["*"],
            sample=None,
            array_join=None,
            join=None,
            prewhere=None,
            where=None,
            having=None,
            group_by=None,
            order_by=None,
            limit_by=None,
            limit=10
        )
        
        assert "LIMIT 10" in sql
