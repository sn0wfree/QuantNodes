# -*- coding: utf-8 -*-
"""DuckDB 节点

支持内存模式和文件模式，支持只读模式
"""
from typing import Optional

import pandas as pd

from QuantNodes.database_node.base import BaseDBNode


class DuckDBNode(BaseDBNode):
    """DuckDB 数据库节点

    支持内存模式和文件模式，支持只读模式

    Args:
        database: 数据库路径，`:memory:` 表示内存模式，
                 绝对路径表示文件模式
        read_only: 是否只读模式（仅文件模式有效，默认 False）

    Example:
        >>> # 内存模式
        >>> node = DuckDBNode(":memory:")

        >>> # 文件模式
        >>> node = DuckDBNode("/data/analysis.duckdb")

        >>> # 只读模式
        >>> node = DuckDBNode("/data/analysis.duckdb", read_only=True)
    """

    def __init__(self, database: str = ":memory:", read_only: bool = False):
        import duckdb
        self._database = database
        self._read_only = read_only
        self._conn = None

    def connect(self):
        """建立 DuckDB 连接"""
        import duckdb
        self._conn = duckdb.connect(self._database, read_only=self._read_only)
        return self._conn

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询"""
        conn = self._conn or self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML"""
        conn = self._conn or self.connect()
        if params:
            result = conn.execute(sql, params)
        else:
            result = conn.execute(sql)
        try:
            return result.rowcount
        except Exception:
            return 0

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        conn = self._conn or self.connect()
        conn.register('temp_df', df)
        if if_exists == 'replace':
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE TABLE {table} AS SELECT * FROM temp_df")
        elif if_exists == 'append':
            try:
                conn.execute(f"INSERT INTO {table} SELECT * FROM temp_df")
            except Exception:
                conn.execute(f"CREATE TABLE {table} AS SELECT * FROM temp_df")
        else:
            conn.execute(f"INSERT INTO {table} SELECT * FROM temp_df")
        conn.unregister('temp_df')
        return len(df)

    def disconnect(self) -> None:
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False
