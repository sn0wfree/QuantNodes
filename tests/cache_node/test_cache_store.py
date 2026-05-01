# coding=utf-8
"""
ParquetCacheStore 单元测试
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from QuantNodes.cache_node.cache_store import ParquetCacheStore


@pytest.fixture
def tmp_cache():
    """临时缓存目录"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def store(tmp_cache):
    return ParquetCacheStore(cache_dir=tmp_cache)


class TestParquetCacheStore:
    """ParquetCacheStore 测试"""

    def test_exists_false(self, store):
        assert not store.exists("test_table")

    def test_write_and_read(self, store):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.write("test_table", df)
        assert store.exists("test_table")

        result = store.read("test_table")
        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_read_nonexistent(self, store):
        result = store.read("nonexistent")
        assert result is None

    def test_write_overwrite(self, store):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [10, 20, 30]})
        store.write("test_table", df1)
        store.write("test_table", df2)

        result = store.read("test_table")
        assert len(result) == 3

    def test_append(self, store):
        df1 = pd.DataFrame({"a": [1, 2], "date": ["2023-01-01", "2023-01-02"]})
        df2 = pd.DataFrame({"a": [3], "date": ["2023-01-03"]})
        store.write("test_table", df1)
        total = store.append("test_table", df2)

        assert total == 3
        result = store.read("test_table")
        assert len(result) == 3

    def test_append_to_empty(self, store):
        df = pd.DataFrame({"a": [1, 2]})
        total = store.append("test_table", df)
        assert total == 2

    def test_delete(self, store):
        df = pd.DataFrame({"a": [1]})
        store.write("test_table", df)
        assert store.exists("test_table")

        result = store.delete("test_table")
        assert result is True
        assert not store.exists("test_table")

    def test_delete_nonexistent(self, store):
        result = store.delete("nonexistent")
        assert result is False

    def test_get_size(self, store):
        assert store.get_size("test_table") == 0

        df = pd.DataFrame({"a": range(100)})
        store.write("test_table", df)
        assert store.get_size("test_table") > 0

    def test_list_tables(self, store):
        assert store.list_tables() == []

        store.write("table_a", pd.DataFrame({"a": [1]}))
        store.write("table_b", pd.DataFrame({"b": [2]}))

        tables = sorted(store.list_tables())
        assert tables == ["table_a", "table_b"]

    def test_table_name_with_dots(self, store):
        """表名中的 . 替换为 __"""
        df = pd.DataFrame({"a": [1]})
        store.write("quote.cn_stock", df)

        table_dir = store._get_table_dir("quote.cn_stock")
        assert table_dir.name == "quote__cn_stock"

    def test_corrupt_parquet(self, store):
        """损坏的 parquet 文件返回 None"""
        table_dir = store._get_table_dir("bad_table")
        table_dir.mkdir(parents=True)
        (table_dir / "data.parquet").write_text("not a parquet file")

        result = store.read("bad_table")
        assert result is None
