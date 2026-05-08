# coding=utf-8
"""
MarketDataCacheNode 集成测试

测试缓存命中、缓存未命中、TTL 过期、强制刷新、增量查询等场景。
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from QuantNodes.cache_node import MarketDataCacheNode, ParquetCacheStore, CacheMetadata
from QuantNodes.cache_node.base import make_cache_key
from QuantNodes.core.node import NodeExecutionError


@pytest.fixture
def tmp_cache():
    d = tempfile.mkdtemp()
    yield d
    if Path(d).exists():
        shutil.rmtree(d)


@pytest.fixture
def mock_db_node():
    """模拟 database_node"""
    node = MagicMock()
    n = 30
    node.query.return_value = pd.DataFrame({
        "ts_code": [f"{i:06d}.SZ" for i in range(n)],
        "trade_date": pd.date_range("2023-07-01", periods=n),
        "close": [10.0 + i for i in range(n)],
        "vol": [1000.0 + i * 100 for i in range(n)],
    })
    return node


class TestMakeCacheKey:
    """cache_key 生成测试"""

    def test_same_inputs_same_key(self):
        k1 = make_cache_key("clickhouse", "quote.cn_stock", ["a", "b"], "WHERE x > 0")
        k2 = make_cache_key("clickhouse", "quote.cn_stock", ["a", "b"], "WHERE x > 0")
        assert k1 == k2

    def test_different_inputs_different_key(self):
        k1 = make_cache_key("clickhouse", "quote.cn_stock", ["a"], "")
        k2 = make_cache_key("clickhouse", "quote.cn_stock", ["a", "b"], "")
        assert k1 != k2

    def test_column_order_independent(self):
        k1 = make_cache_key("clickhouse", "t", ["a", "b"], "")
        k2 = make_cache_key("clickhouse", "t", ["b", "a"], "")
        assert k1 == k2

    def test_key_length(self):
        k = make_cache_key("a", "b", [], "")
        assert len(k) == 12


class TestMarketDataCacheNode:
    """MarketDataCacheNode 测试"""

    def test_cache_miss_full_query(self, tmp_cache, mock_db_node):
        """缓存未命中 → 全量查询"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        result = node.execute({
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        })

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
        mock_db_node.connect.assert_called_once()
        mock_db_node.disconnect.assert_called_once()

    def test_cache_hit(self, tmp_cache, mock_db_node):
        """缓存命中 → 不查询数据库"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        # 第一次: 写入缓存
        input_data = {
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        }
        result1 = node.execute(input_data)
        assert len(result1) == 30
        call_count_1 = mock_db_node.connect.call_count

        # 第二次: 命中缓存, 不再调用数据库
        result2 = node.execute(input_data)
        assert len(result2) == 30
        assert mock_db_node.connect.call_count == call_count_1  # 没有新的连接

    def test_force_refresh(self, tmp_cache, mock_db_node):
        """强制刷新 → 重新查询"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
            "force_refresh": True,
        })

        input_data = {
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        }

        # 第一次
        node.execute(input_data)
        call_count_1 = mock_db_node.connect.call_count

        # 第二次: 强制刷新
        node.execute(input_data)
        assert mock_db_node.connect.call_count > call_count_1

    def test_ttl_expiry(self, tmp_cache, mock_db_node):
        """TTL 过期 → 增量查询"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        input_data = {
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        }

        # 写入缓存
        node.execute(input_data)

        # 手动修改元数据使 TTL 过期
        store = ParquetCacheStore(cache_dir=tmp_cache)
        meta_mgr = CacheMetadata()
        table_dir = store._get_table_dir("quote.cn_stock")
        meta = meta_mgr.load(table_dir)
        meta.created_at = (datetime.now() - timedelta(days=30)).isoformat()
        meta_mgr.save(table_dir, meta)

        # 再次执行: 应该触发增量查询
        result = node.execute(input_data)
        assert isinstance(result, pd.DataFrame)

    def test_invalidate_specific_table(self, tmp_cache, mock_db_node):
        """手动失效指定表"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        node.execute({
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        })

        store = ParquetCacheStore(cache_dir=tmp_cache)
        assert store.exists("quote.cn_stock")

        node.invalidate("quote.cn_stock")
        assert not store.exists("quote.cn_stock")

    def test_invalidate_all(self, tmp_cache, mock_db_node):
        """手动失效所有缓存"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        node.execute({
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        })

        node.invalidate()
        assert not Path(tmp_cache).exists() or not list(Path(tmp_cache).iterdir())

    def test_get_info(self, tmp_cache, mock_db_node):
        """获取缓存状态"""
        node = MarketDataCacheNode(config={
            "cache_dir": tmp_cache,
            "ttl_days": 7,
        })

        node.execute({
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close", "vol"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "node": mock_db_node,
            "date_column": "trade_date",
        })

        info = node.get_info()
        assert "cache_dir" in info
        assert "quote.cn_stock" in info["tables"]
        assert info["tables"]["quote.cn_stock"]["row_count"] == 30

    def test_no_node_raises(self, tmp_cache):
        """缺少 node 参数时抛异常"""
        node = MarketDataCacheNode(config={"cache_dir": tmp_cache})
        with pytest.raises(NodeExecutionError):
            node.execute({
                "source": "clickhouse",
                "table": "test",
                "columns": [],
                "query_filter": "",
            })

    def test_empty_input_raises(self, tmp_cache):
        """空输入时抛异常"""
        node = MarketDataCacheNode(config={"cache_dir": tmp_cache})
        with pytest.raises(NodeExecutionError):
            node.execute(None)
