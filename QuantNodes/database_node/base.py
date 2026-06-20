# -*- coding: utf-8 -*-
"""数据库节点基类

定义所有数据库节点的统一接口
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class BaseDBNode(ABC):
    """数据库节点基类

    所有数据库节点必须实现以下接口：

    Methods:
        connect(): 建立连接
        query(sql, params): 执行查询，返回 DataFrame
        execute(sql, params): 执行 DDL/DML，返回影响行数
        insert_df(df, table, if_exists): 插入 DataFrame
        disconnect(): 关闭连接
        health_check(): 健康检查
    """

    _conn: Any = None

    @abstractmethod
    def connect(self) -> Any:
        """建立数据库连接"""
        raise NotImplementedError

    @abstractmethod
    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行 SQL 查询，返回 DataFrame

        Args:
            sql: SQL 查询语句
            params: 查询参数（可选）

        Returns:
            pd.DataFrame 查询结果
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML 语句

        Args:
            sql: SQL 语句
            params: 语句参数（可选）

        Returns:
            int 影响行数
        """
        raise NotImplementedError

    @abstractmethod
    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """插入 DataFrame 到数据库

        Args:
            df: 要插入的 DataFrame
            table: 目标表名
            if_exists: 表存在时的行为 ('append', 'replace', 'fail')

        Returns:
            int 插入行数
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """关闭数据库连接"""
        raise NotImplementedError

    def health_check(self) -> bool:
        """健康检查

        Returns:
            bool 连接是否正常
        """
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False

    def show_tables(self) -> list:
        """列出所有表。默认通过 SHOW TABLES 实现。

        Returns:
            list[str] 表名列表。

        Note:
            ClickHouse 等需要在 schema 中限定表名的数据库可重写此方法
            (例: ``SHOW TABLES FROM <database>``)。csv/parquet 等无 schema
            概念的 backend 也应重写, 抛 NotImplementedError 或返回自定义列表。
        """
        result = self.query("SHOW TABLES")
        return result.iloc[:, 0].tolist()

    def show_databases(self) -> list:
        """列出所有数据库/Schema。默认通过 SHOW DATABASES 实现。

        Returns:
            list[str] 数据库名列表。

        Note:
            单数据库 backend (sqlite, csv, parquet, 单 instance duckdb)
            可重写返回 ``[self._database]`` 或抛 NotImplementedError。
        """
        result = self.query("SHOW DATABASES")
        return result.iloc[:, 0].tolist()

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False
