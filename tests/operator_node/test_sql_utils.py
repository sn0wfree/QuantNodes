# coding=utf-8
"""QuantNodes.operator_node.sql_utils 单元测试"""
import pytest

from QuantNodes.operator_node.sql_utils import SQLBuilder, TableEngineCreator


class TestTableEngineCreator:
    def test_assemble_cols_2_clause_with_cols(self):
        result = TableEngineCreator._assemble_cols_2_clause("PREFIX", ["col1", "col2"])
        assert result == "PREFIX ( col1,col2 ) "

    def test_assemble_cols_2_clause_without_cols(self):
        result = TableEngineCreator._assemble_cols_2_clause("PREFIX", None, default="DEFAULT")
        assert result == "DEFAULT"

    def test_replacing_merge_tree_creator(self):
        sql = TableEngineCreator.ReplacingMergeTree_creator(
            DB_TABLE="test_table",
            cols_def="col1 Int64, col2 String",
            order_by_cols=["col1"],
        )
        assert "CREATE TABLE" in sql
        assert "test_table" in sql
        assert "ORDER BY" in sql
        assert "ReplacingMergeTree" in sql

    def test_replacing_merge_tree_creator_with_all_options(self):
        sql = TableEngineCreator.ReplacingMergeTree_creator(
            DB_TABLE="test_table",
            cols_def="col1 Int64, col2 String",
            order_by_cols=["col1"],
            sample_by_cols=["col1"],
            partition_by_cols=["col2"],
            primary_by_cols=["col1"],
            ON_CLUSTER="ON CLUSTER 'cluster'",
        )
        assert "SAMPLE BY" in sql
        assert "PARTITION BY" in sql
        assert "PRIMARY BY" in sql
        assert "cluster" in sql

    def test_raw_create_replacing_merge_tree_table_sql(self):
        sql = TableEngineCreator.raw_create_ReplacingMergeTree_table_sql(
            DB_TABLE="test_table",
            cols_def="col1 Int64",
            ORDER_BY_CLAUSE="ORDER BY ( col1 )",
        )
        assert "CREATE TABLE" in sql
        assert "test_table" in sql
        assert "index_granularity" in sql


class TestSQLBuilder:
    def test_assemble_sample_none(self):
        result = SQLBuilder._assemble_sample(None)
        assert result == ""

    def test_assemble_sample_float_less_than_1(self):
        result = SQLBuilder._assemble_sample(0.5)
        assert "SAMPLE" in result
        assert "0.5" in result

    def test_assemble_sample_int(self):
        result = SQLBuilder._assemble_sample(100)
        assert "SAMPLE" in result
        assert "100" in result

    def test_assemble_array_join_none(self):
        result = SQLBuilder._assemble_array_join(None)
        assert result == ""

    def test_assemble_array_join_single(self):
        result = SQLBuilder._assemble_array_join(["arr1"])
        assert "ARRAY JOIN arr1" in result

    def test_assemble_array_join_multiple(self):
        result = SQLBuilder._assemble_array_join(["arr1", "arr2"])
        assert "ARRAY JOIN arr1" in result
        assert "ARRAY JOIN arr2" in result

    def test_assemble_join_none(self):
        result = SQLBuilder._assemble_join(None)
        assert result == ""

    def test_assemble_join_with_info(self):
        result = SQLBuilder._assemble_join({"type": "INNER", "USING": "col1"})
        assert "INNER" in result
        assert "col1" in result

    def test_assemble_where_like_none(self):
        result = SQLBuilder._assemble_where_like(None)
        assert result == ""

    def test_assemble_where_like_single(self):
        result = SQLBuilder._assemble_where_like(["col1 > 0"])
        assert "WHERE" in result
        assert "col1 > 0" in result

    def test_assemble_where_like_multiple(self):
        result = SQLBuilder._assemble_where_like(["col1 > 0", "col2 < 100"])
        assert "col1 > 0 AND col2 < 100" in result

    def test_assemble_group_by_none(self):
        result = SQLBuilder._assemble_group_by(None)
        assert result == ""

    def test_assemble_group_by_single(self):
        result = SQLBuilder._assemble_group_by(["col1"])
        assert "GROUP BY" in result
        assert "col1" in result

    def test_assemble_group_by_multiple(self):
        result = SQLBuilder._assemble_group_by(["col1", "col2"])
        assert "GROUP BY" in result
        assert "col1" in result
        assert "col2" in result

    def test_assemble_order_by_none(self):
        result = SQLBuilder._assemble_order_by(None)
        assert result == ""

    def test_assemble_order_by_single(self):
        result = SQLBuilder._assemble_order_by(["col1"])
        assert "ORDER BY" in result
        assert "col1" in result

    def test_assemble_order_by_multiple(self):
        result = SQLBuilder._assemble_order_by(["col1 DESC", "col2 ASC"])
        assert "ORDER BY" in result

    def test_assemble_limit_by_none(self):
        result = SQLBuilder._assemble_limit_by(None)
        assert result == ""

    def test_assemble_limit_by_with_n(self):
        result = SQLBuilder._assemble_limit_by({"N": 10, "limit_by_cols": ["col1"]})
        assert "LIMIT" in result
        assert "10" in result
        assert "col1" in result

    def test_assemble_limit_none(self):
        result = SQLBuilder._assemble_limit(None)
        assert result == ""

    def test_assemble_limit_with_value(self):
        result = SQLBuilder._assemble_limit(100)
        assert "LIMIT 100" in result

    def test_raw_create_select_sql_basic(self):
        sql = SQLBuilder.raw_create_select_sql(
            SELECT_CLAUSE="col1, col2",
            DB_TABLE="test_table",
        )
        assert "SELECT col1, col2" in sql
        assert "FROM test_table" in sql

    def test_raw_create_select_sql_with_where(self):
        sql = SQLBuilder.raw_create_select_sql(
            SELECT_CLAUSE="col1",
            DB_TABLE="test_table",
            WHERE_CLAUSE="WHERE col1 > 0",
        )
        assert "WHERE col1 > 0" in sql

    def test_raw_create_select_sql_with_subquery(self):
        sql = SQLBuilder.raw_create_select_sql(
            SELECT_CLAUSE="col1",
            DB_TABLE="SELECT col1 FROM inner_table",
        )
        assert "( SELECT col1 FROM inner_table )" in sql

    def test_create_select_sql_basic(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1", "col2"],
        )
        assert "SELECT col1,col2" in sql
        assert "FROM test_table" in sql

    def test_create_select_sql_with_where(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1"],
            where=["col1 > 0"],
        )
        assert "WHERE col1 > 0" in sql

    def test_create_select_sql_with_group_by(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1", "SUM(col2)"],
            group_by=["col1"],
        )
        assert "GROUP BY" in sql
        assert "col1" in sql

    def test_create_select_sql_with_order_by(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1"],
            order_by=["col1 DESC"],
        )
        assert "ORDER BY" in sql
        assert "col1 DESC" in sql

    def test_create_select_sql_with_limit(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1"],
            limit=10,
        )
        assert "LIMIT 10" in sql

    def test_create_select_sql_with_sample(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["col1"],
            sample=0.1,
        )
        assert "SAMPLE" in sql

    def test_create_select_sql_with_join(self):
        sql = SQLBuilder.create_select_sql(
            DB_TABLE="test_table",
            cols=["t1.col1", "t2.col2"],
            join={"type": "LEFT", "using": "id"},
        )
        assert "LEFT" in sql
        assert "USING" in sql
