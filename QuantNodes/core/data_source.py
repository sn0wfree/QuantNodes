# -*- coding: utf-8 -*-
"""DataSource - 顶层数据源标记基类 (Phase 3.3)

QuantNodes 有两个独立的数据接入子树:

  - ``FileFormatLoader`` (文件): 按 path/key 产出 pd.DataFrame
    (H5 / CSV / NPY / Parquet)
  - ``BaseDBNode`` (数据库): 按 SQL 产出 pd.DataFrame
    (SQLite / DuckDB / MySQL / ClickHouse / CSV / Parquet)

两者接口差异大 (文件是面板矩阵语义, 数据库是 SQL 语义), 强行合并会
泄漏抽象。``DataSource`` 因此被设计成**最小化标记基类**, 只统一:

  1. ``close()`` 生命周期 (释放文件句柄 / 数据库连接)
  2. 上下文管理器协议 (``with`` 自动 close)
  3. "产出 pd.DataFrame" 的语义约定 (不强制具体读取签名)

子树各自保留专用接口 (``load`` vs ``query``)。
"""
from abc import ABC, abstractmethod


class DataSource(ABC):
    """所有数据源的顶层标记 + 生命周期协议。

    Subclasses:
        FileFormatLoader: 文件格式适配器 (H5/CSV/NPY/Parquet)
        BaseDBNode: 数据库节点 (SQLite/DuckDB/MySQL/ClickHouse/CSV/Parquet)

    共同契约:
        - 产出 pd.DataFrame (具体读取方法由子树定义)
        - close() 释放底层资源
        - 支持 ``with`` 上下文管理器
    """

    @abstractmethod
    def close(self) -> None:
        """释放底层资源 (文件句柄 / 数据库连接)。"""
        raise NotImplementedError

    def __enter__(self) -> "DataSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
