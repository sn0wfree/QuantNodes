# coding: utf-8
"""database_node 工厂测试 (Phase 3.3)。"""
from __future__ import annotations

import pytest

from QuantNodes.core.data_source import DataSource
from QuantNodes.database_node import (
    SQLiteNode,
    DuckDBNode,
    MySQLNode,
    ClickHouseNode,
    CSVNode,
    ParquetNode,
    create_db_node,
    register_db_node,
    available_sources,
)
from QuantNodes.database_node import factory as factory_mod


@pytest.fixture
def restore_registry():
    """保存/恢复 _DB_NODE_BUILDERS 快照, 隔离 register 测试。"""
    snapshot = dict(factory_mod._DB_NODE_BUILDERS)
    yield
    factory_mod._DB_NODE_BUILDERS.clear()
    factory_mod._DB_NODE_BUILDERS.update(snapshot)


class TestCreateDbNode:
    def test_sqlite(self):
        node = create_db_node("sqlite", database=":memory:")
        assert isinstance(node, SQLiteNode)

    def test_duckdb(self):
        node = create_db_node("duckdb", database=":memory:")
        assert isinstance(node, DuckDBNode)

    def test_mysql(self):
        node = create_db_node("mysql", host="localhost", db="x")
        assert isinstance(node, MySQLNode)

    def test_clickhouse(self):
        node = create_db_node("clickhouse", host="localhost")
        assert isinstance(node, ClickHouseNode)

    def test_csv(self):
        node = create_db_node("csv", filepath="/tmp/x.csv")
        assert isinstance(node, CSVNode)

    def test_parquet(self):
        node = create_db_node("parquet", filepath="/tmp/x.parquet")
        assert isinstance(node, ParquetNode)

    def test_all_nodes_are_datasource(self):
        node = create_db_node("sqlite", database=":memory:")
        assert isinstance(node, DataSource)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unsupported data source"):
            create_db_node("oracle")

    def test_default_params(self):
        node = create_db_node("sqlite")
        assert isinstance(node, SQLiteNode)


class TestAvailableSources:
    def test_lists_builtin_sources(self):
        srcs = available_sources()
        assert set(srcs) == {
            "sqlite", "duckdb", "mysql", "clickhouse", "csv", "parquet",
        }

    def test_sorted(self):
        srcs = available_sources()
        assert srcs == sorted(srcs)


class TestRegisterDbNode:
    def test_register_new(self, restore_registry):
        register_db_node("mem", lambda **p: SQLiteNode(":memory:"))
        assert "mem" in available_sources()
        assert isinstance(create_db_node("mem"), SQLiteNode)

    def test_register_duplicate_raises(self, restore_registry):
        with pytest.raises(ValueError, match="already registered"):
            register_db_node("sqlite", lambda **p: None)

    def test_register_empty_name_raises(self, restore_registry):
        with pytest.raises(ValueError, match="non-empty"):
            register_db_node("", lambda **p: None)
