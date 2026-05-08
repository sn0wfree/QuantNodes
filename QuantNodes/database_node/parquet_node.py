# -*- coding: utf-8 -*-
"""Parquet 读取节点

支持 WHERE 子句过滤
"""
import os
from typing import Optional

import pandas as pd

from QuantNodes.database_node.base import BaseDBNode


class ParquetNode(BaseDBNode):
    """Parquet 文件读取节点

    支持 WHERE 子句过滤

    Args:
        filepath: Parquet 文件绝对路径

    Example:
        >>> node = ParquetNode("/data/users.parquet")
        >>> # 全量读取
        >>> df = node.query()
        >>> # 带 WHERE 过滤
        >>> df = node.query("SELECT * WHERE age > 18")
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._data: Optional[pd.DataFrame] = None

    def connect(self) -> pd.DataFrame:
        """读取 Parquet 到内存"""
        self._data = pd.read_parquet(self._filepath)
        return self._data

    def query(self, sql: Optional[str] = None, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询

        Args:
            sql: SQL 查询语句（可选），支持 WHERE 子句
            params: 查询参数（暂未使用）

        Returns:
            pd.DataFrame 查询结果
        """
        data = self._data or self.connect()

        if sql is None:
            return data

        if 'WHERE' in sql.upper():
            where_clause = sql.split('WHERE', 1)[1].strip()
            return data.query(where_clause)

        return data

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """Parquet 节点不支持 execute"""
        raise NotImplementedError("ParquetNode 不支持 execute 操作")

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """Parquet 节点不支持 insert"""
        raise NotImplementedError("ParquetNode 不支持 insert 操作")

    def disconnect(self) -> None:
        """释放内存"""
        self._data = None

    def health_check(self) -> bool:
        """健康检查"""
        return os.path.exists(self._filepath)
