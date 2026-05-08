# coding=utf-8
"""
MarketDataCacheNode - 行情数据缓存节点

继承 BaseNode, 用 Parquet 文件缓存行情数据。
支持透明代理模式和独立 Pipeline 节点两种集成方式。

工作流程:
1. 生成 cache_key (hash of source+table+columns+filter)
2. 检查缓存是否存在且未过期
3. 命中 → 直接读缓存返回
4. 未命中或过期 → 查询数据源 → 写入缓存 → 返回
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from QuantNodes.core.node import BaseNode, register_node
from QuantNodes.cache_node.cache_store import ParquetCacheStore
from QuantNodes.cache_node.metadata import CacheMetadata


def make_cache_key(source: str, table: str, columns: List[str], query_filter: str) -> str:
    """生成缓存 key (MD5 前 12 位)"""
    key_parts = [
        source,
        table,
        ",".join(sorted(columns or [])),
        query_filter or "",
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


@register_node
class MarketDataCacheNode(BaseNode[Dict[str, Any], pd.DataFrame]):
    """行情数据缓存节点

    透明代理模式:
        在 ConfigBacktestTool._load_from_db() 中自动调用,
        对上层透明, 只需在 DataConfig 中设置 cache_enabled=True。

    独立节点模式:
        在 YAML Pipeline 中作为独立节点使用:
        market_data_cache >> config_executor >> backtest

    输入 (input_data dict):
        source: 数据源类型 (clickhouse/mysql)
        table: 表名
        columns: 列名列表
        query_filter: WHERE 子句
        node: BaseDBNode 实例 (用于回源查询)
        date_column: 日期列名 (用于增量查询)

    输出:
        pd.DataFrame 缓存数据
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or "MarketDataCache", config=config, **kwargs)
        self._store = ParquetCacheStore(
            cache_dir=self.config.get("cache_dir", "~/.quantnodes/cache")
        )
        self._meta = CacheMetadata()
        self._ttl_days = self.config.get("ttl_days", 7)
        self._force_refresh = self.config.get("force_refresh", False)

    def _execute(self, input_data: Dict[str, Any] = None, **kwargs) -> pd.DataFrame:
        """执行缓存查询

        1. 生成 cache_key
        2. 检查缓存 → 命中则直接返回
        3. 未命中 → 查询数据源 → 写缓存 → 返回
        """
        if input_data is None:
            raise ValueError("input_data is required")

        source = input_data["source"]
        table = input_data["table"]
        columns = input_data.get("columns", [])
        query_filter = input_data.get("query_filter", "")
        db_node = input_data.get("node")
        date_column = input_data.get("date_column", "")

        cache_key = make_cache_key(source, table, columns, query_filter)
        table_dir = self._store._get_table_dir(table)

        # 1. 检查缓存
        if not self._force_refresh and self._store.exists(table):
            cached_meta = self._meta.load(table_dir)
            if cached_meta is not None and not self._meta.is_expired(cached_meta):
                # 缓存命中
                self._meta.touch(cached_meta)
                self._meta.save(table_dir, cached_meta)
                df = self._store.read(table)
                if df is not None:
                    return df

        # 2. 缓存未命中或过期 → 查询数据源
        if db_node is None:
            raise ValueError("input_data['node'] (BaseDBNode) is required for cache miss")

        # 3. 尝试增量查询
        if self._store.exists(table) and date_column:
            cached_meta = self._meta.load(table_dir)
            if cached_meta is not None and cached_meta.date_range:
                last_date = cached_meta.date_range[1]
                if last_date:
                    new_df = self._incremental_query(
                        db_node, source, table, columns,
                        query_filter, date_column, last_date
                    )
                    if new_df is not None and len(new_df) > 0:
                        total = self._store.append(table, new_df)
                        self._update_meta_after_write(
                            table, cache_key, source, query_filter,
                            columns, date_column, table_dir
                        )
                        return self._store.read(table)

        # 4. 全量查询
        df = self._full_query(db_node, source, table, columns, query_filter)

        # 5. 写入缓存
        if df is not None and not df.empty:
            self._store.write(table, df)
            self._update_meta_after_write(
                table, cache_key, source, query_filter,
                columns, date_column, table_dir, df
            )

        return df if df is not None else pd.DataFrame()

    def _full_query(
        self, db_node, source: str, table: str,
        columns: List[str], query_filter: str
    ) -> Optional[pd.DataFrame]:
        """全量查询"""
        cols_str = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols_str} FROM {table}"
        if query_filter:
            sql += " " + query_filter.lstrip("WHERE ").lstrip("where ")

        db_node.connect()
        try:
            return db_node.query(sql)
        finally:
            db_node.disconnect()

    def _incremental_query(
        self, db_node, source: str, table: str,
        columns: List[str], query_filter: str,
        date_column: str, last_date: str
    ) -> Optional[pd.DataFrame]:
        """增量查询: 只查新增数据"""
        cols_str = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols_str} FROM {table}"

        # 构建 WHERE 子句
        where_parts = []
        if query_filter:
            where_parts.append(query_filter.lstrip("WHERE ").lstrip("where "))

        # 使用 db_date_column (数据库原始列名) 来做增量查询
        # 这里 date_column 可能是映射后的名字, 需要从 query_filter 中推断原始列名
        # 简单方案: 直接用日期比较
        where_parts.append(f"{date_column} > '{last_date}'")

        sql += " WHERE " + " AND ".join(where_parts)

        db_node.connect()
        try:
            return db_node.query(sql)
        finally:
            db_node.disconnect()

    def _update_meta_after_write(
        self, table: str, cache_key: str, source: str,
        query_filter: str, columns: List[str],
        date_column: str, table_dir: Path,
        df: pd.DataFrame = None,
    ) -> None:
        """写入后更新元数据"""
        if df is None:
            df = self._store.read(table)
        if df is None:
            return

        date_range = []
        if date_column and date_column in df.columns:
            date_range = [
                str(df[date_column].min()),
                str(df[date_column].max()),
            ]

        meta = self._meta.create(
            table=table,
            cache_key=cache_key,
            source=source,
            query_filter=query_filter,
            ttl_days=self._ttl_days,
            row_count=len(df),
            columns=list(df.columns),
            date_range=date_range,
        )
        self._meta.save(table_dir, meta)

    def invalidate(self, table: str = None) -> None:
        """手动失效缓存

        Args:
            table: 指定表名, 为 None 时清除所有缓存
        """
        if table:
            self._store.delete(table)
        else:
            import shutil
            if self._store.cache_dir.exists():
                shutil.rmtree(self._store.cache_dir)

    def get_info(self) -> Dict[str, Any]:
        """获取缓存状态信息"""
        tables = self._store.list_tables()
        info = {
            "cache_dir": str(self._store.cache_dir),
            "tables": {},
        }
        for t in tables:
            size = self._store.get_size(t)
            table_dir = self._store._get_table_dir(t)
            meta = self._meta.load(table_dir)
            info["tables"][t] = {
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 2),
                "expired": self._meta.is_expired(meta) if meta else True,
                "row_count": meta.row_count if meta else 0,
                "created_at": meta.created_at if meta else None,
            }
        return info
