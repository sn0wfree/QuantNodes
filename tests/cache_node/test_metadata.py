# coding=utf-8
"""
CacheMetadata 单元测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from QuantNodes.cache_node.metadata import CacheMetadata, CacheMeta


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def meta():
    return CacheMetadata()


class TestCacheMetadata:
    """CacheMetadata 测试"""

    def test_save_and_load(self, meta, tmp_dir):
        m = CacheMeta(
            table="test",
            cache_key="abc123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            ttl_days=7,
            row_count=100,
            columns=["a", "b"],
            date_range=["2023-01-01", "2023-12-31"],
            source="clickhouse",
            query_filter="WHERE x > 0",
        )
        meta.save(tmp_dir, m)
        loaded = meta.load(tmp_dir)

        assert loaded is not None
        assert loaded.table == "test"
        assert loaded.cache_key == "abc123"
        assert loaded.row_count == 100
        assert loaded.columns == ["a", "b"]
        assert loaded.date_range == ["2023-01-01", "2023-12-31"]

    def test_load_nonexistent(self, meta, tmp_dir):
        result = meta.load(tmp_dir)
        assert result is None

    def test_load_corrupt(self, meta, tmp_dir):
        (tmp_dir / "metadata.json").write_text("not json {{{")
        result = meta.load(tmp_dir)
        assert result is None

    def test_is_expired_false(self, meta):
        m = CacheMeta(
            created_at=datetime.now().isoformat(),
            ttl_days=7,
        )
        assert not meta.is_expired(m)

    def test_is_expired_true(self, meta):
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        m = CacheMeta(
            created_at=old_time,
            ttl_days=7,
        )
        assert meta.is_expired(m)

    def test_is_expired_no_created_at(self, meta):
        m = CacheMeta(created_at="")
        assert meta.is_expired(m)

    def test_touch(self, meta):
        m = CacheMeta(last_accessed="2020-01-01T00:00:00")
        meta.touch(m)
        assert m.last_accessed != "2020-01-01T00:00:00"

    def test_create(self, meta):
        m = meta.create(
            table="quote.cn_stock",
            cache_key="abc123",
            source="clickhouse",
            query_filter="WHERE x > 0",
            ttl_days=7,
            row_count=1000,
            columns=["a", "b", "c"],
            date_range=["2023-01-01", "2023-12-31"],
        )
        assert m.table == "quote.cn_stock"
        assert m.row_count == 1000
        assert m.ttl_days == 7
        assert m.created_at  # should be set
