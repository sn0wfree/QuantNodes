# -*- coding: utf-8 -*-
"""database_node 工厂 (Phase 3.3)

将"按 source 字符串选择数据库后端"的逻辑收敛到单一注册表驱动工厂,
取代散落在调用方 (如 ``agent/tools/config_backtest.py``) 的 if/elif 阶梯。

Usage:
    from QuantNodes.database_node import create_db_node

    node = create_db_node("sqlite", database="/data/x.db")
    node = create_db_node("clickhouse", host="localhost", database="default")

    # 扩展新后端
    from QuantNodes.database_node import register_db_node
    register_db_node("myback", lambda **p: MyBackendNode(**p))

注意: 工厂只负责 "参数 -> 实例"; 连接参数的来源 (conn.ini / path 解析)
仍由调用方负责。
"""
from typing import Callable, Dict

from QuantNodes.database_node.base import BaseDBNode
from QuantNodes.database_node.sqlite_node import SQLiteNode
from QuantNodes.database_node.duckdb_node import DuckDBNode
from QuantNodes.database_node.mysql_node import MySQLNode
from QuantNodes.database_node.clickhouse_node import ClickHouseNode
from QuantNodes.database_node.csv_node import CSVNode
from QuantNodes.database_node.parquet_node import ParquetNode

DBNodeBuilder = Callable[..., BaseDBNode]

_DB_NODE_BUILDERS: Dict[str, DBNodeBuilder] = {
    "sqlite": lambda **p: SQLiteNode(**p),
    "duckdb": lambda **p: DuckDBNode(**p),
    "mysql": lambda **p: MySQLNode(**p),
    "clickhouse": lambda **p: ClickHouseNode(**p),
    "csv": lambda **p: CSVNode(**p),
    "parquet": lambda **p: ParquetNode(**p),
}


def create_db_node(source: str, **params) -> BaseDBNode:
    """按 source 字符串创建对应的数据库节点实例。

    Args:
        source: 后端类型, 见 ``available_sources()``。
        **params: 透传给对应 Node 构造函数的关键字参数。

    Returns:
        BaseDBNode 子类实例。

    Raises:
        ValueError: source 未注册。
    """
    builder = _DB_NODE_BUILDERS.get(source)
    if builder is None:
        raise ValueError(
            f"Unsupported data source: {source}. "
            f"Available: {sorted(_DB_NODE_BUILDERS)}"
        )
    return builder(**params)


def register_db_node(source: str, builder: DBNodeBuilder) -> None:
    """注册新的后端 builder (供扩展)。

    Args:
        source: 后端类型字符串。
        builder: 接收 **params 返回 BaseDBNode 实例的可调用对象。

    Raises:
        ValueError: source 为空或已存在。
    """
    if not source:
        raise ValueError("source must be a non-empty string")
    if source in _DB_NODE_BUILDERS:
        raise ValueError(f"source '{source}' already registered")
    _DB_NODE_BUILDERS[source] = builder


def available_sources() -> list:
    """返回已注册的后端类型列表 (排序)。"""
    return sorted(_DB_NODE_BUILDERS)
