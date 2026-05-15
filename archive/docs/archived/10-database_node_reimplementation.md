# database_node 重新实现设计文档

> **状态**: Draft  
> **创建日期**: 2026-04-27  
> **目标**: 将 `database_node/` 打造为数据库节点统一入口

---

## 1. 现状分析

### 1.1 现有代码位置

| 来源 | 文件 | 行数 | 功能状态 | 决策 |
|------|------|------|---------|------|
| `utils_node` | `MySQLConn_v004_node.py` | 464 | 功能完整，含连接池 | ✅ 移动+适配 |
| `utils_node` | `ch2pandas_node.py` | 371 | HTTP ClickHouse | ✅ 移动+适配 |
| `utils_node` | `ch2pandas.py` | 236 | HTTP ClickHouse (简化版) | ❌ 废弃 |
| `database_node` | `sqlite3_node.py` | 33 | 缺 pandas 导入 | ✅ 修复 |
| `database_node` | `clickhouse.py` | 0 | 空文件 | ❌ 删除 |
| `database_node` | `mysql_node.py` | 164 | 全注释 | ❌ 删除 |
| `database_node` | `Redisconnector.py` | 136 | 全注释 | ❌ 删除 |
| `database_node` | `__init__.py` | 4 | 导出错误 | ✅ 重写 |

### 1.2 pyproject.toml 现有依赖

```toml
[tool.poetry.dependencies]
# 数据库层
SQLAlchemy = "^2.0.0"        # MySQL 连接池
clickhouse-connect = "^0.6.0" # ClickHouse 官方 driver
PyMySQL = "^1.0.0"           # MySQL
duckdb = "^0.9.0"            # DuckDB
pandas = "^2.1.0"            # 数据处理
```

**无需新增依赖**，所有所需依赖已存在于 `pyproject.toml`。

---

## 2. 设计目标

### 2.1 功能需求

| 节点 | 功能 | 数据源 | 内存模式 | 文件模式 | 连接池 | WHERE 过滤 |
|------|------|--------|---------|---------|---------|-----------|
| `SQLiteNode` | 查询/插入 | `.db` | ✅ | ✅ | ❌ | ❌ |
| `DuckDBNode` | 查询/插入/分析 | `.duckdb` | ✅ | ✅ | ❌ | ❌ |
| `MySQLNode` | 查询/插入/DDL | 远程服务器 | ❌ | ✅ | ✅ 可配置 | ❌ |
| `ClickHouseNode` | 查询/插入/DDL | 远程服务器 | ❌ | ✅ | ✅ 可配置 | ❌ |
| `CSVNode` | 读取/过滤 | `.csv` | ❌ | ✅ | ❌ | ✅ |
| `ParquetNode` | 读取/过滤 | `.parquet` | ❌ | ✅ | ❌ | ✅ |

### 2.2 统一接口设计

```python
from QuantNodes.core.quant_nodes_object import QuantNodesObject
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseDBNode(QuantNodesObject):
    """数据库节点基类 - 统一接口
    
    所有数据库节点必须实现以下接口：
    
    Methods:
        connect(): 建立连接
        query(sql, params): 执行查询，返回 DataFrame
        execute(sql, params): 执行 DDL/DML，返回影响行数
        insert_df(df, table, if_exists): 插入 DataFrame
        disconnect(): 关闭连接
        health_check(): 健康检查
    """
    
    def connect(self) -> Any:
        """建立数据库连接"""
        raise NotImplementedError
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行 SQL 查询，返回 DataFrame
        
        Args:
            sql: SQL 查询语句
            params: 查询参数（可选）
            
        Returns:
            pd.DataFrame 查询结果
        """
        raise NotImplementedError
    
    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DDL/DML 语句
        
        Args:
            sql: SQL 语句
            params: 语句参数（可选）
            
        Returns:
            int 影响行数
        """
        raise NotImplementedError
    
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
    
    def disconnect(self) -> None:
        """关闭数据库连接"""
        raise NotImplementedError
    
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool 连接是否正常
        """
        raise NotImplementedError
```

---

## 3. 详细实施方案

### 3.1 文件结构

```
database_node/
├── __init__.py              # 统一导出
├── base.py                  # BaseDBNode 基类 (~50 行)
├── sqlite_node.py           # SQLite (~100 行) ← 修复
├── duckdb_node.py          # DuckDB (~120 行) ← 新建
├── mysql_node.py           # MySQL (~250 行) ← 移动+适配
├── clickhouse_node.py      # ClickHouse (~250 行) ← 移动+适配
├── csv_node.py             # CSV (~80 行) ← 新建
└── parquet_node.py         # Parquet (~80 行) ← 新建
```

### 3.2 SQLiteNode 实现

```python
# -*- coding: utf-8 -*-
"""SQLite 节点

支持内存模式和文件模式
"""
import sqlite3
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
        self._conn = None
    
    def connect(self) -> sqlite3.Connection:
        """建立 SQLite 连接"""
        self._conn = sqlite3.connect(self._database)
        return self._conn
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行查询"""
        conn = self._conn or self.connect()
        return pd.read_sql(sql, conn, params=params)
    
    def execute(self, sql: str, params: tuple = None) -> int:
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
```

### 3.3 DuckDBNode 实现

```python
# -*- coding: utf-8 -*-
"""DuckDB 节点

支持内存模式和文件模式，支持只读模式
"""
import duckdb
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
        self._database = database
        self._read_only = read_only
        self._conn = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """建立 DuckDB 连接"""
        self._conn = duckdb.connect(self._database, read_only=self._read_only)
        return self._conn
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行查询"""
        conn = self._conn or self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()
    
    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DDL/DML"""
        conn = self._conn or self.connect()
        if params:
            result = conn.execute(sql, params)
        else:
            result = conn.execute(sql)
        return result.rowcount
    
    def insert_df(self, df: pd.DataFrame, table: str, 
                 if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        conn = self._conn or self.connect()
        conn.register('temp_df', df)
        if if_exists == 'replace':
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM temp_df")
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
```

### 3.4 MySQLNode 实现

**来源**: `utils_node/MySQLConn_v004_node.py`（移动+适配）

```python
# -*- coding: utf-8 -*-
"""MySQL 节点

支持连接池，基于 pymysql + SQLAlchemy
"""
import pandas as pd
import pymysql
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
        self._conn = None
    
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
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行查询"""
        engine = self._engine or self.connect()
        return pd.read_sql(sql, engine, params=params)
    
    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DDL/DML"""
        engine = self._engine or self.connect()
        with engine.connect() as conn:
            result = conn.execute(sql, params or ())
            conn.commit()
            return result.rowcount
    
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
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False
    
    # 保留原有 MySQLConn 的有用方法
    def show_tables(self) -> list:
        """列出所有表"""
        result = self.query("SHOW TABLES")
        return result.iloc[:, 0].tolist()
    
    def show_databases(self) -> list:
        """列出所有数据库"""
        result = self.query("SHOW DATABASES")
        return result.iloc[:, 0].tolist()
```

### 3.5 ClickHouseNode 实现

**来源**: `utils_node/ch2pandas_node.py`（移动+适配）

```python
# -*- coding: utf-8 -*-
"""ClickHouse 节点

支持 HTTP 接口和官方 driver 双接口
"""
import pandas as pd
from QuantNodes.database_node.base import BaseDBNode


class ClickHouseNode(BaseDBNode):
    """ClickHouse 数据库节点
    
    支持 HTTP 接口和官方 driver 双接口
    
    Args:
        host: 主机地址
        port: 端口 (HTTP 默认 8123，Native 默认 9000)
        user: 用户名 (默认 default)
        passwd: 密码
        database: 数据库名 (默认 default)
        interface: 接口类型 ('http' 或 'native'，默认 'http')
        pool_size: 连接池大小 (默认 10，可配置)
    
    Example:
        >>> # HTTP 接口
        >>> node = ClickHouseNode(
        ...     host="localhost",
        ...     user="default",
        ...     passwd="",
        ...     database="default"
        ... )
        
        >>> # Native 接口
        >>> node = ClickHouseNode(
        ...     host="localhost",
        ...     port=9000,
        ...     interface="native"
        ... )
    """
    
    def __init__(self, host: str, port: int = 8123,
                 user: str = 'default', passwd: str = '',
                 database: str = 'default',
                 interface: str = 'http',
                 pool_size: int = 10):
        self._host = host
        self._port = port
        self._user = user
        self._passwd = passwd
        self._database = database
        self._interface = interface
        self._pool_size = pool_size
        self._client = None
    
    def connect(self):
        """建立连接"""
        if self._interface == 'native':
            import clickhouse_connect
            self._client = clickhouse_connect.get_client(
                host=self._host,
                port=self._port,
                username=self._user,
                password=self._passwd,
                database=self._database,
            )
        else:
            # HTTP 接口 - 基于现有 CHBase 实现
            self._client = CHBase(
                name=self._database,
                user=self._user,
                passwd=self._passwd,
                host=self._host,
                port=self._port,
                db=self._database,
            )
        return self._client
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行查询"""
        client = self._client or self.connect()
        if self._interface == 'native':
            return client.query(sql).result_rows
        else:
            return client.get(sql, convert_to='DataFrame')
    
    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DDL/DML"""
        client = self._client or self.connect()
        if self._interface == 'native':
            client.command(sql)
            return 0
        else:
            client.insert_query(sql)
            return 0
    
    def insert_df(self, df: pd.DataFrame, table: str, 
                 if_exists: str = 'append') -> int:
        """插入 DataFrame"""
        client = self._client or self.connect()
        full_table = f"{self._database}.{table}"
        if self._interface == 'native':
            client.insert_df(table, df)
        else:
            client.insert(df, self._database, table)
        return len(df)
    
    def disconnect(self) -> None:
        """关闭连接"""
        if self._client:
            if self._interface == 'native':
                self._client.close()
            self._client = None
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.query("SELECT 1")
            return True
        except Exception:
            return False
    
    def show_tables(self) -> list:
        """列出所有表"""
        result = self.query(f"SHOW TABLES FROM {self._database}")
        return result.iloc[:, 0].tolist()
    
    def show_databases(self) -> list:
        """列出所有数据库"""
        result = self.query("SHOW DATABASES")
        return result.iloc[:, 0].tolist()
```

### 3.6 CSVNode 实现

```python
# -*- coding: utf-8 -*-
"""CSV 读取节点

支持 WHERE 子句过滤
"""
import os
import pandas as pd
from QuantNodes.database_node.base import BaseDBNode


class CSVNode(BaseDBNode):
    """CSV 文件读取节点
    
    支持 WHERE 子句过滤
    
    Args:
        filepath: CSV 文件绝对路径
        encoding: 字符编码 (默认 utf-8)
        sep: 分隔符 (默认 ,)
    
    Example:
        >>> node = CSVNode("/data/users.csv")
        >>> # 全量读取
        >>> df = node.query()
        >>> # 带 WHERE 过滤
        >>> df = node.query("SELECT * WHERE age > 18")
    """
    
    def __init__(self, filepath: str, encoding: str = 'utf-8', sep: str = ','):
        self._filepath = filepath
        self._encoding = encoding
        self._sep = sep
        self._data = None
    
    def connect(self):
        """读取 CSV 到内存"""
        self._data = pd.read_csv(
            self._filepath, 
            encoding=self._encoding, 
            sep=self._sep
        )
        return self._data
    
    def query(self, sql: str = None, params: tuple = None) -> pd.DataFrame:
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
        
        # 解析 WHERE 子句
        if 'WHERE' in sql.upper():
            where_clause = sql.split('WHERE', 1)[1].strip()
            return data.query(where_clause)
        
        return data
    
    def execute(self, sql: str, params: tuple = None) -> int:
        """CSV 节点不支持 execute"""
        raise NotImplementedError("CSVNode 不支持 execute 操作")
    
    def insert_df(self, df: pd.DataFrame, table: str, 
                 if_exists: str = 'append') -> int:
        """CSV 节点不支持 insert"""
        raise NotImplementedError("CSVNode 不支持 insert 操作")
    
    def disconnect(self) -> None:
        """释放内存"""
        self._data = None
    
    def health_check(self) -> bool:
        """健康检查"""
        return os.path.exists(self._filepath)
```

### 3.7 ParquetNode 实现

```python
# -*- coding: utf-8 -*-
"""Parquet 读取节点

支持 WHERE 子句过滤
"""
import os
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
        self._data = None
    
    def connect(self):
        """读取 Parquet 到内存"""
        self._data = pd.read_parquet(self._filepath)
        return self._data
    
    def query(self, sql: str = None, params: tuple = None) -> pd.DataFrame:
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
        
        # 解析 WHERE 子句
        if 'WHERE' in sql.upper():
            where_clause = sql.split('WHERE', 1)[1].strip()
            return data.query(where_clause)
        
        return data
    
    def execute(self, sql: str, params: tuple = None) -> int:
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
```

---

## 4. 导出设计

```python
# database_node/__init__.py
"""
database_node - 数据库节点统一入口

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

__all__ = [
    'BaseDBNode',
    'SQLiteNode',
    'DuckDBNode',
    'MySQLNode',
    'ClickHouseNode',
    'CSVNode',
    'ParquetNode',
]
```

---

## 5. 依赖管理

### 5.1 现有依赖（无需新增）

```toml
[tool.poetry.dependencies]
# 数据库层
SQLAlchemy = "^2.0.0"        # MySQL 连接池
clickhouse-connect = "^0.6.0" # ClickHouse 官方 driver
PyMySQL = "^1.0.0"           # MySQL
duckdb = "^0.9.0"            # DuckDB
pandas = "^2.1.0"            # 数据处理
pyarrow = ">=14.0.0"         # Parquet 支持（需确认是否已存在）
```

### 5.2 需要确认的依赖

| 依赖 | 状态 | 说明 |
|------|------|------|
| `pyarrow` | ❓ 需确认 | Parquet 读取需要 |
| `clickhouse-driver` | ❓ 需确认 | ClickHouse Native 接口需要 |

---

## 6. 实施步骤

| # | 步骤 | 文件 | 工作量 | 优先级 | 状态 |
|---|------|------|--------|---------|------|
| 1 | 创建 `base.py` | BaseDBNode 基类 | ~1h | 高 | ⏳ |
| 2 | 修复 `sqlite_node.py` | 添加 import，完善接口 | ~0.5h | 高 | ⏳ |
| 3 | 新建 `duckdb_node.py` | DuckDB 实现 | ~1h | 中 | ⏳ |
| 4 | 移动/适配 `mysql_node.py` | MySQL 实现 | ~1.5h | 高 | ⏳ |
| 5 | 移动/适配 `clickhouse_node.py` | ClickHouse 实现 | ~1.5h | 高 | ⏳ |
| 6 | 新建 `csv_node.py` | CSV 实现 | ~0.5h | 中 | ⏳ |
| 7 | 新建 `parquet_node.py` | Parquet 实现 | ~0.5h | 中 | ⏳ |
| 8 | 更新 `__init__.py` | 统一导出 | ~0.25h | 中 | ⏳ |
| 9 | 删除废弃文件 | clickhouse.py, mysql_node.py, Redisconnector.py | ~0.25h | 低 | ⏳ |
| 10 | 语法检查 | 所有文件 | ~0.5h | 高 | ⏳ |

**总计**: ~7.5h

---

## 7. 讨论问题确认

### ✅ 已确认

| # | 问题 | 决策 |
|---|------|------|
| 1 | MySQL/ClickHouse 连接池大小 | ✅ 可配置 (`pool_size` 参数) |
| 2 | DuckDB 只读模式 | ✅ 需要 (`read_only` 参数) |
| 3 | CSV/Parquet WHERE 过滤 | ✅ 需要 (解析 WHERE 子句) |
| 4 | 错误处理/重连机制 | ✅ OK (按默认策略) |

### ❓ 待确认

| # | 问题 | 建议 |
|---|------|------|
| 1 | `pyarrow` 是否已在 `pyproject.toml` 中？ | 需要添加 `pyarrow>=14.0.0` |
| 2 | `clickhouse-driver` 是否已在 `pyproject.toml` 中？ | 需要添加 `clickhouse-driver>=0.2.0` |
| 3 | 是否需要保留 `utils_node/` 中的旧实现？ | 建议标记为废弃，后续删除 |

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 路径含空格导致工具读取困难 | 中 | 实施时注意路径处理 |
| `utils_node/` 旧代码依赖 | 低 | 移动后更新导入路径 |
| 连接池配置不当 | 低 | 提供合理默认值 |
| CSV/Parquet WHERE 解析 | 中 | 使用 pandas.query() 而非完整 SQL 解析 |

---

**最后更新**: 2026-04-27
