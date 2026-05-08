# coding=utf-8
"""
Parquet 缓存存储引擎

用 Parquet 文件缓存行情数据, 支持读写/追加/删除/存在检查。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ParquetCacheStore:
    """Parquet 文件缓存存储

    目录结构:
        cache_dir/
        └── {table}/
            └── data.parquet

    table 中的 '.' 自动替换为 '__' (quote.cn_stock → quote__cn_stock)
    """

    def __init__(self, cache_dir: str = "~/.quantnodes/cache"):
        self.cache_dir = Path(cache_dir).expanduser()

    def _get_table_dir(self, table: str) -> Path:
        """表名 → 缓存目录"""
        safe_name = table.replace(".", "__")
        return self.cache_dir / safe_name

    def _get_data_path(self, table: str) -> Path:
        """表名 → data.parquet 路径"""
        return self._get_table_dir(table) / "data.parquet"

    def exists(self, table: str) -> bool:
        """检查缓存是否存在"""
        return self._get_data_path(table).exists()

    def read(self, table: str) -> Optional[pd.DataFrame]:
        """读取缓存, 不存在则返回 None"""
        path = self._get_data_path(table)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def write(self, table: str, df: pd.DataFrame) -> None:
        """写入缓存 (覆盖模式)"""
        table_dir = self._get_table_dir(table)
        table_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(table_dir / "data.parquet", index=False)

    def append(self, table: str, df_new: pd.DataFrame) -> int:
        """追加数据到缓存

        自动去重: 以 df_new 的行追加到已有数据。
        返回追加后的总行数。
        """
        existing = self.read(table)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            combined = df_new

        self.write(table, combined)
        return len(combined)

    def delete(self, table: str) -> bool:
        """删除缓存, 返回是否成功"""
        table_dir = self._get_table_dir(table)
        if not table_dir.exists():
            return False
        import shutil
        shutil.rmtree(table_dir)
        return True

    def get_size(self, table: str) -> int:
        """获取缓存文件大小 (bytes), 不存在返回 0"""
        path = self._get_data_path(table)
        if not path.exists():
            return 0
        return path.stat().st_size

    def list_tables(self):
        """列出所有缓存的表"""
        if not self.cache_dir.exists():
            return []
        tables = []
        for d in self.cache_dir.iterdir():
            if d.is_dir() and (d / "data.parquet").exists():
                tables.append(d.name.replace("__", "."))
        return tables
