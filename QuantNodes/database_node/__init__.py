# -*- coding: utf-8 -*-
"""database_node - 数据库节点统一入口

支持 SQLite, DuckDB, MySQL, ClickHouse, CSV, Parquet 等数据源

Usage:
    from QuantNodes.database_node import (
        SQLiteNode,
        DuckDBNode,
        MySQLNode,
        ClickHouseNode,
        CSVNode,
        ParquetNode,
    )

    # SQLite 内存模式
    sqlite = SQLiteNode(":memory:")
    sqlite.connect()

    # DuckDB 文件模式
    duckdb = DuckDBNode("/data/analysis.duckdb")

    # MySQL 带连接池
    mysql = MySQLNode(
        host="localhost",
        user="root",
        passwd="password",
        db="mydb",
        pool_size=20
    )

    # ClickHouse HTTP 接口
    ch = ClickHouseNode(
        host="localhost",
        user="default",
        passwd="",
        database="default"
    )

    # CSV 读取
    csv = CSVNode("/data/users.csv")
    df = csv.query("SELECT * WHERE age > 18")
"""

from QuantNodes.database_node.base import BaseDBNode
from QuantNodes.database_node.sqlite_node import SQLiteNode
from QuantNodes.database_node.duckdb_node import DuckDBNode
from QuantNodes.database_node.mysql_node import MySQLNode
from QuantNodes.database_node.clickhouse_node import ClickHouseNode
from QuantNodes.database_node.csv_node import CSVNode
from QuantNodes.database_node.parquet_node import ParquetNode
from QuantNodes.database_node.factory import (
    create_db_node,
    register_db_node,
    available_sources,
)

__all__ = [
    'BaseDBNode',
    'SQLiteNode',
    'DuckDBNode',
    'MySQLNode',
    'ClickHouseNode',
    'CSVNode',
    'ParquetNode',
    'create_db_node',
    'register_db_node',
    'available_sources',
]
