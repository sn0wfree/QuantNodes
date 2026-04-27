# -*- coding: utf-8 -*-
"""SQLite 节点

支持内存模式和文件模式
"""
import sqlite3
from typing import Optional

import pandas as pd

from QuantNodes.database_node.base import BaseDBNode


class SQLiteNode(BaseDBNode):
    """SQLite 数据库节点

    支持内存模式 (`:memory:`) 和文件模式 (绝对路径)

    Args:
        database: 数据库路径，`:memory:` 表示内存模式，
                 绝对路径表示文件模式

    Example:
        >>> # 内存模式
        >>> node = SQLiteNode(":memory:")
        >>> node.connect()
        >>> node.execute("CREATE TABLE test (id INT, name TEXT)")
        >>> node.query("SELECT * FROM test")

        >>> # 文件模式
        >>> node = SQLiteNode("/data/mydb.sqlite")
    """

    def __init__(self, database: str = ":memory:"):
        self._database = database
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """建立 SQLite 连接"""
        self._conn = sqlite3.connect(self._database)
        return self._conn

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询"""
        conn = self._conn or self.connect()
        return pd.read_sql(sql, conn, params=params)

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML"""
        conn = self._conn or self.connect()
        cur = conn.cursor()
        cur.execute(sql, params or ())
        conn.commit()
        return cur.rowcount

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        df.to_sql(table, self._conn or self.connect(),
                  if_exists=if_exists, index=False)
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
