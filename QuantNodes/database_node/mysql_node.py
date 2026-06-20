# -*- coding: utf-8 -*-
"""MySQL 节点

支持连接池，基于 pymysql + SQLAlchemy
"""
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, pool

from QuantNodes.database_node.base import BaseDBNode


class MySQLNode(BaseDBNode):
    """MySQL 数据库节点

    支持连接池

    Args:
        host: 主机地址
        port: 端口 (默认 3306)
        user: 用户名
        passwd: 密码
        db: 数据库名
        charset: 字符集 (默认 UTF8)
        pool_size: 连接池大小 (默认 10，可配置)
        pool_recycle: 连接回收时间秒 (默认 3600，可配置)

    Example:
        >>> node = MySQLNode(
        ...     host="localhost",
        ...     user="root",
        ...     passwd="password",
        ...     db="mydb",
        ...     pool_size=20
        ... )
        >>> node.query("SELECT * FROM users LIMIT 10")
    """

    def __init__(self, host: str, port: int = 3306,
                 user: str = '', passwd: str = '', db: str = '',
                 charset: str = 'UTF8',
                 pool_size: int = 10,
                 pool_recycle: int = 3600):
        self._host = host
        self._port = port
        self._user = user
        self._passwd = passwd
        self._db = db
        self._charset = charset
        self._pool_size = pool_size
        self._pool_recycle = pool_recycle
        self._engine = None

    def _build_url(self) -> str:
        """构建 SQLAlchemy URL"""
        return (
            f"mysql+pymysql://{self._user}:{self._passwd}"
            f"@{self._host}:{self._port}/{self._db}"
            f"?charset={self._charset}&local_infile=1"
        )

    def connect(self):
        """建立连接（返回 SQLAlchemy Engine）"""
        self._engine = create_engine(
            self._build_url(),
            pool_size=self._pool_size,
            pool_recycle=self._pool_recycle,
            poolclass=pool.QueuePool,
        )
        return self._engine

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询"""
        engine = self._engine or self.connect()
        return pd.read_sql(sql, engine, params=params)

    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML"""
        engine = self._engine or self.connect()
        with engine.connect() as conn:
            result = conn.execute(sql, params or ())
            conn.commit()
            try:
                return result.rowcount
            except Exception:
                return 0

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        engine = self._engine or self.connect()
        df.to_sql(table, engine, if_exists=if_exists, index=False)
        return len(df)

    def disconnect(self) -> None:
        """关闭连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
